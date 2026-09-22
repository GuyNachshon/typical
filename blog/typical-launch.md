---
title: "Typical: the decision primitive, self-hosted"
date: 2026-09-22
---

# Typical: the decision primitive, self-hosted

Most software eventually needs a small decision made against unstructured input: route this ticket, flag this post, size this alert. For a while the only tool for that was a full LLM call: a prompt, a wait, a block of text to parse, and a confidence score that nobody trained to mean anything.

TypeSafe shipped something built for exactly this gap. Jev is a "System-One model": you give it a state and a typed question (Choice, Score, or Noul, their names for pick-one, ordinal, and yes/no) and it returns calibrated probabilities directly, no generated text to parse. It landed. A benchmark followed (JevBench), and behind the benchmark came a wave of people building the same idea in the open: SemIf, OpenJev, system-one-open, Laya, jeff, open-jev-deberta, more than forty entries on the public leaderboard at last count.

Jev itself stayed closed. It's an API: $0.0399 per 1,000 decisions, 0.65 s p50, on TypeSafe's servers, under TypeSafe's rate limits, with no published weights and no way to run it yourself. That's a defensible business. It's also a strange place to put a primitive whose entire pitch is that it belongs inside your control flow, next to your database calls, not behind a network round-trip you rent by the token.

Typical is the same idea, self-hosted. Apache-2.0 weights over an Apache-2.0 backbone, inference code and training recipe in the repo, the full experimental report (every run, every dead end, every cost) public, and it runs on one GPU you already own at 15-21 ms per decision with no per-call charge. Three primitives, matching Jev's own vocabulary: **Choice**, **Score**, **Noul**.

## Choice: pick one of K labels you define right now

```python
from typical import Typical

m = Typical.from_pretrained("OzLabs/typical-small", device="auto")

m.choice(state, "What does the customer want?", ["refund", "replacement", "repair"])
# -> {"refund": p, "replacement": p, "repair": p, "∅": p_null}
```

The label list isn't a fixed classifier head. It's plain text, supplied at call time, and the model was never trained on this specific set of three words. That's what replaces: an LLM call, a JSON parser hoping the model picked one of your options and not a fourth one it invented, and a "confidence" field that was usually the model grading its own homework.

The number to check isn't a demo, it's whether Choice generalizes to label sets it has never seen. On CLINC-150 (151 intents), `typical-small` scores .801 and `typical-medium` .847; on 20 Newsgroups, a completely different label vocabulary, .540 and .588; on held-out workflow families it never trained on directly, .818 and .865 (`RESULTS.md` §1, §4). None of these label sets were fixed at training time. The model reads the strings you give it.

## Score: ordinal levels, where being close matters

```python
m.score(state, "How urgent is this ticket?", ["0", "1", "2", "3"])
# -> {"0": p, "1": p, "2": p, "3": p, "∅": p_null}
```

A plain K-way classifier treats "urgent" and "not urgent" as two unrelated buckets, exactly as wrong when it says 0 instead of 1 as when it says 0 instead of 3. Score is trained with ordinal-smoothed targets instead, so nearby levels share probability mass and being one level off costs less than being three levels off.

The isolated test: on held-out urgency ratings, switching a K-way Choice head to ordinal-smoothed Score targets took NLL from 2.07 to 1.23 and ordinal MAE from .58 to .55, with the exact same top-1 predictions on every ordinal item in JevBench itself, meaning the accuracy didn't move and the probabilities got honest (`REPORT.md` §3ac). On an external severity-rating set (systemone-lite, hard tier) the same swap took accuracy from .853 to .939 and MAE from .15 to .06. `typical-small` and `typical-medium` ship with this head on; held-out score NLL is 1.01 and 1.03 respectively, and 0.95 for the (unreleased) 14B candidate (`RESULTS.md` §3).

## Noul: yes or no, and the order can't matter

```python
m.noul(state, "Is the order still under warranty?")
# -> p_yes
```

A two-way Choice over `["yes", "no"]` is still a softmax over rendered options, which means it can shift its answer if you happen to list "no" first. Noul isn't that: it's a dedicated Bernoulli head reading the decision state directly, with no candidate text rendered at all. There's nothing for order to act on.

We checked this on the shipped `typical-small` checkpoint: reversing the label order leaves `P(yes)` exactly unchanged on 10 of 11 held-out test sets, and off by 0.009 on the eleventh (`releases/typical-small.md`). Held-out noul accuracy is .710 (`typical-small`), .811 (`typical-medium`), and .831 for the 14B candidate (`RESULTS.md` §3). In the isolated ablation that established the design, switching from a 2-way Choice to the Bernoulli head took an external, never-trained-on floor test (PagerDuty, constant-prediction floor .792) from .602 to .886 (`REPORT.md` §3ac). The shipped checkpoints don't all clear that particular external floor yet (more on that under Limitations), but the head design itself is doing what it was built to do.

## Abstention is a fourth answer, not a fifth label

Every one of the three primitives can also say "none of these fit." That's not a label you add to the list and hope the model remembers to use correctly: it's architecturally separate, a dedicated head decision computed independently of the softmax over your real options, so it doesn't compete for probability mass with them.

That separation is what makes `p_null` usable operationally: a low value means answer, a value crossing some threshold you pick means escalate to a human or a slower model, and the threshold can be tuned per deployment because the number means the same thing regardless of how many candidates you passed in. In the architecture ablation that established this design, null-detection AUROC across a K-sweep from 5 to 150 candidates stayed in a tight .92-.98 band on CLINC-150 and .69-.90 on Banking77 (`REPORT.md` §3t). Before the fix, the same checkpoint always abstained at K=150 and never abstained at K=5: a null score that means something different at every K is not usable as a threshold at all.

## How it works

Encode the state (a document, a ticket, a conversation) through the backbone once, and cache it as a KV cache. For each question, render its candidates as a short suffix and read that suffix's terminal hidden state, `h_D`, back against the cached prefix: no re-encoding the state, no query attending to any other query. A small head turns `h_D` plus the rendered option spans into a probability over your candidates union `{∅}`.

<p align="center"><img src="../figures/fig_architecture.png" alt="Typical architecture: state encoded once into a KV cache, per-question suffixes read out a calibrated distribution" width="720"></p>
<p align="center"><em>Figure: one state encode, many cheap per-question reads against the cached prefix.</em></p>

The backbone is a truncated Qwen3 (or Qwen3.5) base model, mostly frozen, with a LoRA adapter (rank 16) on its top layers doing the adaptation. Training mixes four weighted families in every batch: evidence tasks (NLI), knowledge MCQ, workflow decisions, and a soft-target uncertainty corpus. The workflow bucket includes a hard curriculum called DecisionMix v2, where the same state and candidates get rendered under two or more different rubrics with different gold answers, so the model has to read the rubric rather than pattern-match it. The three typed heads described above sit on top of the same backbone and share the same state encoding. Full derivation, every ablation, and the pre-registered pass/fail rules are in [`REPORT.md`](../REPORT.md); this is the readable version.

## The numbers

<p align="center"><img src="../figures/fig_ladder.png" alt="JevBench standard and hard accuracy across backbone size, Qwen3 vs Qwen3.5, frozen vs trained, with dotted reference lines for the rest of the leaderboard" width="640"></p>
<p align="center"><em>Figure: JevBench standard and hard accuracy across the size ladder, against the rest of the leaderboard (dotted lines).</em></p>

| model | backbone | JevBench standard | JevBench hard | warm p50 latency | status |
|---|---|---|---|---|---|
| `typical-small` | Qwen3-1.7B-Base | .694 | .432 | 15.5-17 ms | released |
| `typical-medium` | Qwen3-4B-Base | .806 | .423 | 19-21 ms | released |
| `typical-large` candidate (`tl1b`, 14B) | Qwen3-14B-Base | .931 | .450 | 59 ms (in-process, pre-serving-optimization) | trained, not released |

The 14B is trained and evaluated but not released: before training it, we set a pass rule (hard-tier accuracy ≥ .559 or hard-tier Brier ≤ .65, long-document policy questions ≥ .35), and it misses both: narrowly on Brier (.66 against .65), by a wide margin on long-document policy accuracy (.158). We'd rather hold a checkpoint back than ship one that doesn't clear its own bar.

On the public subset of JevBench (the harness's 231 public items; a further 146 judge items exist but aren't public, so treat any gap under a couple of points as noise at this sample size):

| entry | standard | hard |
|---|---|---|
| `typical-small` | .694 | .432 |
| `typical-medium` | .806 | .423 |
| `typical-large` candidate (`tl1b`) | .931 | .450 |
| open-jev-deberta-v3-large | .431 | .378 |
| Laya (ModernBERT) | .694 | .351 |
| jeff (GLiFormer 400M) | .750 | .387 |
| system-one-open (Gemma-E2B LoRA) | .931 | .486 |
| SemIf (Qwen3.5-4B) | .986 | .613 |
| OpenJev (26B-A4B) | .972 | .640 |
| Jev 1.13.0 (closed) | .986 | .730 |

Calibration: in the isolated ablation, ordinal-smoothed Score targets took held-out score NLL from 2.07 to 1.23 with zero decisions changed (`REPORT.md` §3ac). At full scale, the 14B candidate (which bundles the typed heads with a long-state fix, a Brier term and calibration-based checkpoint selection) posts a held-out score NLL of 0.95, against 2.87 for the same backbone under the prior recipe (`REPORT.md` §3ah). We also trained that recipe with the frozen-teacher distillation switched off, changing one flag and nothing else: it came out *better* on every hard-tier number (hard .477 against .450, long-document policy .211 against .158) and on validation NLL. The teacher we added to buy hard reasoning bought none of it (`REPORT.md` §3ai).

<p align="center"><img src="../figures/fig_calibration.png" alt="Held-out score NLL and typed-decisions NLL across checkpoints, with the ladder_14b-to-tl1b jump annotated" width="640"></p>
<p align="center"><em>Figure: held-out score and typed-decisions NLL across checkpoints. The long-state fix and calibration changes cut the 14B's score NLL from 2.87 to 0.95; a matched control shows the frozen-teacher distillation contributed none of it.</em></p>

Latency, against what a normal LLM call costs depending on how it has to answer:

<p align="center"><img src="../figures/fig_latency_quality.png" alt="Single-decision latency vs JevBench standard accuracy for the Typical family, plotted against the latency bands for one-letter decode, JSON or label decode, and chain-of-thought" width="640"></p>
<p align="center"><em>Figure: the Typical family's latency vs. accuracy, next to the latency bands a normal LLM call falls into depending on how it answers.</em></p>

## The hard tier is where a single forward pass runs out

Every model in this project, at every size, does worse on JevBench's hard tier than on standard: long-document policy questions, multi-step composition, temporal and unit reasoning. `typical-medium`'s hard-tier accuracy is .423; the 14B candidate's is .450. That's the project's clearest open problem, and it's a data and objective problem, not obviously a capacity one: a frozen 14B backbone given three examples in its prompt, with no training at all, scores .559 on the same hard tier (`REPORT.md` §3ah). Training on our current recipe buys standard-tier accuracy and calibration, and does not yet buy the hard tier.

<p align="center"><img src="../figures/fig_hard_families.png" alt="JevBench hard-tier accuracy by family for ladder_14b, tl1b, and the frozen 14B with three examples" width="640"></p>
<p align="center"><em>Figure: hard-tier accuracy by family. Long-document policy and multi-step composition drag the average down independently.</em></p>

What we're doing about it: a calibration objective built for this tier specifically (log-loss plus a Brier term plus ordinal structure, reported per family instead of averaged into one number); more generator work on the composition families (temporal, unit conversion, trade-off reasoning) where nothing we've tried has moved the needle; and the long-state fix below, which is necessary but not sufficient on its own.

## What we learned building this

**Abstention needs to be architecture, not vocabulary.** Our first attempt at "none of the above" rendered it as a literal option in the label list, same as everything else. It behaved like a landmine: at 150 candidates the model always abstained, at 5 it never did, and reversing which candidate came first swung standard-tier accuracy by 4 points. Making abstention a separate head decision, computed independently of the softmax over the real labels, fixed the pathology outright (`REPORT.md` §3t). If you're adding a "none of these" option to any classifier, check whether it's a candidate or a decision. It should be the second one.

**A silent truncation bug taught a model to be confidently wrong about long documents.** Our long-policy training rows put the facts at the end of the text; the training pipeline truncates state to a fixed token budget from the front. At the 1,024-token budget we shipped with, that cut the facts off before the model ever saw them in 99% of those rows: we were training it to answer confidently about content it never read. We caught it by the bug's signature (very confident, very wrong, only on long inputs), then fixed it two ways: regenerate the data facts-first, and refuse to truncate. Drop any row that doesn't fit rather than silently cutting it (`REPORT.md` §3ag). If a model gets more confidently wrong as its input gets longer, check the data loader's token limit before you check the model.

**Fine-tuning can make the hardest cases worse than not fine-tuning at all.** That's the 14B result above, restated as a general warning: training buys accuracy and calibration everywhere the training distribution actually covers the task, and can cost both exactly where it doesn't. A frozen model with a few examples in context isn't a strawman to beat before you ship; it's a ceiling to check you've actually cleared.

**A quarter to a third of our serving latency was one deep copy.** The serving path cloned the entire cached KV state on every decision: correct, safe, and much slower than it needed to be. Swapping the clone for a zero-copy view (bit-identical output, verified) cut warm p50 from 21-25 ms to 15.5-17 ms on the 1.7B model and 26-27 ms to 19-21 ms on the 4B (`runs/serve_bench2/results.json`; `REPORT.md` §3ag). Before reaching for a bigger GPU or a smaller model, profile what your serving code does with state you're supposed to be reusing.

<p align="center"><img src="../figures/fig_truncation.png" alt="Left: state token length distribution against truncation cutoffs at 256/1024/2048/3072 tokens. Right: long-policy accuracy for ladder_14b, typical-medium, tl1b, and the frozen 14B" width="640"></p>
<p align="center"><em>Figure: left, how much of a long-policy row gets truncated at each token budget. Right, long-policy accuracy after the fix.</em></p>

<p align="center"><img src="../figures/fig_serving.png" alt="Cold and warm p50 latency before and after removing the per-decision KV cache deep copy, for typical-small, typical-medium, and tm2" width="640"></p>
<p align="center"><em>Figure: removing the deep copy took roughly a quarter to a third off warm p50 latency, across the family.</em></p>

## Limitations, plainly

- **Hard tier**, covered above, is the biggest open item.
- **Level-7 composition** (temporal, unit, expected-value, and trade-off reasoning combined) sits at roughly chance for every checkpoint we've trained; the generator that would teach it doesn't exist yet.
- **Very large candidate sets** (K in the hundreds to thousands) need a different serving path (an energy-score front end feeding a top-r shortlist into the native head), not the native head directly, because the native suffix has a practical token budget.
- **External floors are mixed.** Noul's Bernoulli head clears an external severity-rating floor by a wide margin in the isolated ablation that built it; the shipped checkpoints don't all clear every external floor we test against yet, and we report the numbers as they are rather than the ones that flatter the launch.
- **Order sensitivity remains on Choice** (not Noul): reordering rendered candidates moves the answer by a few points, a letter-interface artifact no rendering scheme has fully removed.

## Try it

```bash
pip install -r inference/requirements.txt
```

```python
from typical import Typical

m = Typical.from_pretrained("OzLabs/typical-small", device="auto")  # or "typical-medium"

m.choice(state, "What does the customer want?", ["refund", "replacement", "repair"])
m.noul(state, "Is the order still under warranty?")
m.score(state, "How urgent is this ticket?", ["0", "1", "2", "3"])
```

Or click before you code:

```bash
uv run --no-sync python demo/app.py
```

Pick one decision your software currently makes by calling an LLM and parsing the answer, write down the label set as it actually varies at runtime, and swap in `m.choice()` or `m.noul()` with your real state and labels. If the model abstains a lot, that's telling you something about your label set. If accuracy is worse than a frozen larger model would give you, believe that number before you believe ours.

What's next: the hard-tier calibration objective, more work on the composition families, a `typical-large` release once the 14B clears its own pass rule, and a Qwen3.5 port already in progress (a Qwen3.5-4B checkpoint posts the best hard-tier number of any size we control, .495, not yet released either). Models: [`OzLabs/typical-small-preview`](https://huggingface.co/OzLabs/typical-small-preview), [`OzLabs/typical-small`](https://huggingface.co/OzLabs/typical-small), [`OzLabs/typical-medium`](https://huggingface.co/OzLabs/typical-medium). Full report: [`REPORT.md`](../REPORT.md). Comparison notes: [`COMPARE.md`](../COMPARE.md).
