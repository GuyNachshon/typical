---
title: "Typical: calibrated decisions in one forward pass"
date: 2026-09-22
---

# Typical: calibrated decisions in one forward pass

Somewhere in your product, code has to make a decision. A support ticket comes in and something has to pick refund, replacement, or repair. A moderation queue fills up and something has to say whether a post crosses a line. An alert fires and something has to rate how bad it is, on a scale someone made up in a design doc.

The default way to build that "something" today is to call an LLM: write a prompt, ask for JSON back, wait a few hundred milliseconds to several seconds, then parse whatever came out, hoping it isn't wrapped in a markdown fence, doesn't add a caveat before the JSON, and picked one of the labels you actually offered. If it hands you a confidence number, that number was invented after the fact, usually by asking the model to grade itself, and it means whatever it decided it should mean that day.

That's a text-generation problem wearing a decision problem's clothes. Typical strips the text generation back out.

## What Typical is

Typical is a decision head, not a chatbot. You give it three things: a **state** (the ticket, the document, the conversation so far), a **question**, and a **label set you define at call time**. It gives back a calibrated probability over exactly those labels, plus an explicit "none of the above," in one forward pass with no sampling and no completion to parse.

```python
from typical import Typical

m = Typical.from_pretrained("OzLabs/typical-small", device="auto")

m.choice(state, "What does the customer want?", ["refund", "replacement", "repair"])
# -> {"refund": p, "replacement": p, "repair": p, "∅": p_null}
```

There are three primitives, and they cover most of what "decide something" means in practice:

- **Choice**: pick one of K labels you supply at inference time. The model was never trained on your label set; it reads the strings and scores them.
- **Score**: an ordinal version of Choice (severity 1-5, urgency low/medium/high), trained with ordinal-smoothed targets so being one level off costs less than being three levels off.
- **Noul**: yes/no, through a dedicated Bernoulli head rather than a two-way Choice. That matters more than it sounds: a Choice head over `["yes", "no"]` can and does shift its answer if you list the options in the other order. The Bernoulli head, by construction, doesn't render the options at all, so it can't. Measured on the shipped `typical-small` checkpoint: reversing the label order leaves `P(yes)` exactly unchanged on 10 of 11 held-out sets, and off by 0.009 on the eleventh.

A single decision runs in 15-21 ms depending on model size, warm, on one GPU (`runs/serve_bench2/results.json`; see the latency section below). No token generation, so nothing to parse and nothing that can come back malformed.

## Why it works

Ask a normal LLM a multiple-choice question and it does something wasteful: it computes an internal representation of "the answer is B," then spends a full autoregressive step turning that representation into the token "B," which you then have to detokenize and match back against your label list. The interesting number, the model's actual belief about B versus A versus C, existed as a hidden state one layer before any of that started. Typical reads that layer instead of generating from it: a small trained head, plus a thin LoRA adapter on its top layers, sits on a truncated, mostly-frozen Qwen3 backbone and turns the state-after-seeing-the-candidates into a real probability distribution directly.

Two consequences follow from doing it this way instead of by prompting:

**The state gets encoded once, and questions are cheap after that.** The document, ticket, or conversation goes through the model exactly once and gets cached as a KV cache. Every question after that is a short suffix attending back into that cache, without re-encoding or re-reading the state. On a 1.7B model, adding a second question to an already-encoded state costs about 3 ms marginally at K=2; even at K=256 candidates it's 28 ms marginally, not another full pass (`RESULTS.md` §6). Ask one question or twenty against the same ticket and the state cost is paid exactly once.

**Abstention is a real output, not a fifth label you asked the model to remember to use.** "None of these labels fit" is architecturally separate from the label softmax: a dedicated head decision, not a candidate competing for probability mass with your real options. Rendering "none of the above" as a literal option in the list, we found, is a bug factory (more on that below).

## The numbers, including the ones we'd rather not have

<p align="center"><img src="../figures/fig_architecture.png" alt="Typical architecture: state encoded once into a KV cache, per-question suffixes read out a calibrated distribution" width="720"></p>
<p align="center"><em>Figure: one state encode, many cheap per-question reads against the cached prefix.</em></p>

The released family, on the public subset of JevBench (a decision-focused benchmark; more on what "public subset" means in a second):

| model | backbone | JevBench standard | JevBench hard | warm p50 latency | status |
|---|---|---|---|---|---|
| `typical-small` | Qwen3-1.7B-Base | .694 | .432 | 15.5-17 ms | released |
| `typical-medium` | Qwen3-4B-Base | .806 | .423 | 19-21 ms | released |
| `typical-large` candidate (`tl1b`) | Qwen3-14B-Base | **.931** | .450 | 59 ms (in-process, pre-serving-optimization) | candidate, not released |

<p align="center"><img src="../figures/fig_ladder.png" alt="JevBench standard and hard accuracy across backbone size, Qwen3 vs Qwen3.5, frozen vs trained, with dotted reference lines for the rest of the leaderboard" width="640"></p>
<p align="center"><em>Figure: JevBench standard and hard accuracy across the size ladder, against the rest of the leaderboard (dotted lines). 4B is the knee of the Qwen3 line; 14B is the ceiling we haven't shipped yet.</em></p>

A word on "JevBench standard": the harness has 231 public decision items across standard, easy, and hard tiers, plus 146 held-out judge items we don't have access to. We run the 231 public ones, a real evaluation but not a ranked leaderboard entry, since the harness only ranks runs with ≥95% coverage including the private items. Chance on this split is .311 standard / .284 easy / .336 hard, with an effective sample size around 36 on the standard tier, so treat any gap under a couple of points between our own checkpoints as noise (`COMPARE.md` §4a).

Against the rest of the field on those same 231 items, we sit above untrained classifiers (a DeBERTa-v3 baseline scores .431 standard), roughly level with `system-one-open` (a Gemma-E2B LoRA at .931 standard, .486 hard), and behind the leaders: SemIf and OpenJev (open, .97-.99 standard, .61-.64 hard) and the closed Jev 1.13.0 (.986 standard, .730 hard). We're not claiming the top of that board. We're claiming a specific, useful spot on it: open weights, a documented recipe, and a latency number the closed leaders don't publish.

<p align="center"><img src="../figures/fig_latency_quality.png" alt="Single-decision latency vs JevBench standard accuracy for the Typical family, plotted against the latency bands for one-letter decode, JSON or label decode, and chain-of-thought" width="640"></p>
<p align="center"><em>Figure: the Typical family's latency vs. accuracy, next to what a normal LLM call costs depending on how it answers.</em></p>

Calibration is where the typed heads earn their keep. In an isolated ablation on the 1.7B backbone, switching Score from a plain K-way Choice to ordinal-smoothed targets cuts held-out score NLL from 2.07 to 1.23, with zero decisions changed on the 12 ordinal items in JevBench itself (`REPORT.md` §3ac). At full scale, the `typical-large` candidate checkpoint (which bundles the typed heads with the fixes described below) posts a held-out score NLL of 0.95 against 2.87 for the same backbone under the prior recipe (`REPORT.md` §3ah). That's a probability you can put a threshold on, versus one you can't.

<p align="center"><img src="../figures/fig_calibration.png" alt="Held-out score NLL and typed-decisions NLL across checkpoints, with the ladder_14b-to-tl1b jump annotated" width="640"></p>
<p align="center"><em>Figure: held-out score and typed-decisions NLL across checkpoints. The long-state fix plus frozen-teacher distillation cut the 14B's score NLL from 2.87 to 0.95.</em></p>

**The 14B checkpoint (`tl1b`) is not released.** It posts the best standard-tier number in the project, .931, level with the best open entry on the leaderboard. But we set a pass rule before training it (hard-tier accuracy ≥ .559 or hard-tier Brier ≤ .65, long-document policy questions ≥ .35), and it misses on both: narrowly on Brier (.66 against a .65 bar), and by a wide margin on long-document accuracy (.158). Worse, the same 14B backbone, frozen, with three examples in the prompt and no training at all, scores .559 on the hard tier, higher than our trained model's .450, and better calibrated (Brier .60 vs .66). Training this model made it more confident and more accurate on the easy stuff, and worse than doing nothing on the hard stuff. We'd rather tell you that than ship it.

<p align="center"><img src="../figures/fig_hard_families.png" alt="JevBench hard-tier accuracy by family for ladder_14b, tl1b, and the frozen 14B with three examples" width="640"></p>
<p align="center"><em>Figure: hard-tier accuracy by family. Long-document policy and multi-step composition drag the average down independently, and the frozen 14B beats both trained checkpoints on most of them.</em></p>

## What we learned that you can use even if you never touch this model

**Abstention needs to be architecture, not vocabulary.** Our first attempt at "none of the above" rendered it as a literal option in the label list, same as everything else. It behaved like a landmine: at 150 candidates the model always abstained, at 5 it never did, and reversing which candidate came first swung standard-tier accuracy by 4 points. Making abstention a separate head decision (computed independently of the softmax over your labels, not one more thing competing for probability mass) fixed the pathology outright; abstention accuracy became monotone in the number of candidates (`REPORT.md` §3t). If you're building a "none of these" option into any classifier, check whether it's a candidate or a decision. It should be the second one.

**A silent truncation bug taught a model to be confidently wrong about long documents.** Our long-policy training rows put the facts at the *end* of the text; the pipeline truncates state to a fixed token budget from the front. At the 1,024-token budget we shipped with, that cut the facts off before the model ever saw them in 99% of those rows: we were training it to answer confidently about content it never read. We caught it by the bug's signature (very confident, very wrong, only on long inputs), then fixed it two ways: regenerate the data facts-first, and refuse to truncate. Drop any row that doesn't fit rather than silently cutting it (`REPORT.md` §3ag). If a model gets more confidently wrong as its input gets longer, check the data loader's token limit before you check the model.

**Fine-tuning can make the hardest cases worse than not fine-tuning at all.** That's the 14B result above, restated as a warning: training buys accuracy and calibration everywhere your distribution actually covers the task, and can cost both exactly where it doesn't. And "where it doesn't" tends to concentrate in whatever you'd call the hard tier. A frozen model with a few examples in context isn't a strawman to beat before you ship; it's a ceiling to check you've actually cleared.

**A quarter to a third of our serving latency was one deep copy.** The serving path cloned the entire cached KV state on every decision: correct, safe, and much slower than it needed to be. Swapping the clone for a zero-copy view (bit-identical output, verified) cut warm p50 from 21-25 ms to 15.5-17 ms on the 1.7B model and 26-27 ms to 19-21 ms on the 4B (`runs/serve_bench2/results.json`; `REPORT.md` §3ag). Before reaching for a bigger GPU or a smaller model, profile what your serving code does with state you're supposed to be reusing.

<p align="center"><img src="../figures/fig_truncation.png" alt="Left: state token length distribution against truncation cutoffs at 256/1024/2048/3072 tokens. Right: long-policy accuracy for ladder_14b, typical-medium, tl1b, and the frozen 14B" width="640"></p>
<p align="center"><em>Figure: left, how much of a long-policy row gets truncated at each token budget (98% of rows lost their facts at 1,024 tokens). Right, long-policy accuracy after the fix: ladder_14b .05 to tl1b .16, still short of the frozen 14B's .42.</em></p>

<p align="center"><img src="../figures/fig_serving.png" alt="Cold and warm p50 latency before and after removing the per-decision KV cache deep copy, for typical-small, typical-medium, and tm2" width="640"></p>
<p align="center"><em>Figure: one zero-copy view instead of a deep copy took roughly a quarter to a third off warm p50 latency, across every model in the family.</em></p>

## Two-minute quickstart

Download the `inference/` folder from either model's Hugging Face repo: a self-contained package with no dependency on our training code, just `torch`, `transformers`, `safetensors`, `huggingface_hub`, and `numpy`.

```bash
pip install -r inference/requirements.txt
```

```python
from typical import Typical

m = Typical.from_pretrained("OzLabs/typical-small", device="auto")  # or "typical-medium"

# K-way choice over a label set you define right here
m.choice(state, "What does the customer want?", ["refund", "replacement", "repair"])

# Yes/no, through the order-invariant Bernoulli head
m.noul(state, "Is the order still under warranty?")

# Ordinal levels, with a real expected value
m.score(state, "How urgent is this ticket?", ["0", "1", "2", "3"])
```

`state` is a string or a JSON-serializable dict. If you'd rather click before you code, clone the repo and run the local demo, which downloads whichever checkpoint you pick and lets you try your own state, question, and labels in a browser:

```bash
uv run --no-sync python demo/app.py
```

## Roadmap, and an invitation

Next up, in order: a calibration objective built for the hard tier specifically (log-loss plus a Brier term plus ordinal structure, reported per family instead of averaged away); more work on the composition families (temporal reasoning, unit conversion, trade-offs), where no amount of data we've thrown at it has moved the needle yet; a `typical-large` release once the 14B checkpoint actually clears its own pass rule; a port to the Qwen3.5 family (already working: a Qwen3.5-4B checkpoint posts the best hard-tier number of any size we control, .495, though it isn't released yet either); and a manual per-shape CUDA graph capture path to bring warm latency down further (naive `torch.compile` got to 8 ms on one shape and then produced different probabilities on the others, so it's on hold until it's done right).

Everything above is reproducible: the models are on Hugging Face (`OzLabs/typical-small-preview`, `OzLabs/typical-small`, `OzLabs/typical-medium`), the full report with every experiment, dead end, and cost is in [`REPORT.md`](../REPORT.md), and the head-to-head against the closed alternative is in [`COMPARE.md`](../COMPARE.md).

If you want to try it on your own data: pick one decision your software currently makes by calling an LLM and parsing the answer, write down the label set as it actually varies at runtime, and swap in `m.choice()` or `m.noul()` with your real state and your real labels. If the model abstains a lot, that's it telling you something about your label set, not a bug to route around. If accuracy is worse than a frozen larger model would give you, believe that number before you believe ours.
