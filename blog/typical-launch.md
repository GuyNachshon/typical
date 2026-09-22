---
title: "Typical: Models That Decide, Not Generate"
date: 2026-09-22
---

# Typical: Models That Decide, Not Generate

Most AI calls inside software don't need prose. A router needs one destination. A policy check needs yes or no. An incident system needs a severity. An agent needs its next action.

We usually build all of those out of a model trained to generate text:

```
prompt -> tokens -> parser -> decision
```

Every stage after the first is overhead. The model writes a sentence you throw away, your parser hopes it picked one of your options and not a fourth one it invented, and the "confidence" field that comes back was the model grading its own homework.

Typical has a different interface:

```
state + typed question -> probability distribution
```

No generated answer. Nothing to parse. Today we're releasing **Typical Small** (1.7B) and **Typical Medium** (4B), open-weight decision models for three primitives: Choice, Noul, and Score. Those three names, and the shape of the interface, come from TypeSafe's Jev, which established this class of model; ours is an open implementation of the same primitive, not a new one.

## Show me

```python
from typical import Typical

m = Typical.from_pretrained("OzLabs/typical-small", device="auto")

state = ("Ticket #7734: A customer writes: 'I was charged twice for my last order "
         "and the tracking number you sent doesn't work. I need this fixed today.'")

m.choice(state, "Which team should own this ticket?",
         ["billing", "shipping", "support", "retention"])
```

```python
{'billing': 0.0466, 'shipping': 0.3454, 'support': 0.5845, 'retention': 0.0234,
 'p_null': 0.0841}
```

That is a real return value from the released 1.7B checkpoint, not an illustration — it's the recorded pass behind the trace on our site. The label list is plain text you supply at call time. The model was never trained on this particular set of four words, and there is no classifier head with `billing` baked into index 0. The output space belongs to your program, not to the model.

Note the two parts. The four labels are a distribution conditioned on one of them being right; `p_null` is a separate head's answer to whether any of them is, and it doesn't compete with them for mass.

## Three decision primitives

- **Choice** — pick one of K options defined at runtime.
- **Noul** — return P(yes) for a proposition.
- **Score** — return a distribution over ordered levels.

```python
m.choice(state, "What does the customer want?", ["refund", "replacement", "repair"])
m.noul(state,   "Is the order still under warranty?")
m.score(state,  "How urgent is this ticket?", ["0", "1", "2", "3"])
```

Each is a different head on the same backbone, and the difference is structural.

Choice reads your label strings, so it can operate over label sets that were never hard-coded into an output head. On CLINC-150 (151 intents) `typical-small` scores .801 and `typical-medium` .847 — though CLINC is in the training mix, so read that as a capacity number, not as evidence about unseen labels.

The evidence about unseen labels is the sets we held out. On CLINC intents that were removed from training entirely, and whose label strings the models have never scored, .932 and .937. On 20 Newsgroups, held out as a dataset and an entirely different label vocabulary, .540 and .588. On held-out workflow families neither model trained on, .836 and .874. The first of those is the claim: the label list is read at call time, so intents that didn't exist during training still work.

Noul is not `Choice(["yes", "no"])`. It has a dedicated Bernoulli head with no candidate text rendered at all, so there is no candidate order for the answer to depend on. On the shipped `typical-small` checkpoint, reversing label order leaves P(yes) exactly unchanged on 10 of the 11 held-out test sets and moves it by .009 on the eleventh — that one is the set whose labels aren't literally `yes`/`no`, so it routes to Choice, which is order-sensitive. Call `noul()` with the proposition and you get the invariant path. Held-out noul accuracy is .715 for Small and .811 for Medium.

Score knows adjacent levels are related. A severity of 2 is closer to 3 than to 0, so we train it with ordinal-smoothed targets rather than treating levels as unrelated buckets. In the isolated ablation that cut held-out score NLL from 2.07 to 1.23 on a four-level set — where a model that knows nothing scores ln 4 = 1.39, so the change moved the probabilities from worse than uniform to better than it — without changing a single top-1 prediction on the JevBench ordinal items.

## Abstention is a decision, not a label

All three primitives can say "none of these fit," and that answer comes from a separate head rather than an extra entry in your label list.

The difference is not cosmetic. When we rendered "none of the above" as just another option, abstention became a function of how many candidates you passed: at 150 it always abstained, at 5 it never did. A null score that means something different at every K cannot be thresholded, which kills the thing you wanted it for, namely routing the uncertain cases to a human or a slower model. Moving abstention into its own head removed the pathology outright.

## How it works

Typical reads the state once.

The state (a ticket, a document, an agent trace) becomes a cached model prefix. Each decision is then a short question scored against that same prefix. Instead of decoding an answer token, Typical reads the hidden states of the question and its candidates directly and turns them into probabilities. Adding a second question to the same state costs one short forward pass, not a second read of the document.

<p align="center"><img src="../figures/fig_architecture.png" alt="Typical architecture: state encoded once into a KV cache, per-question suffixes read out a probability distribution" width="720"></p>
<p align="center"><em>One state encode, many cheap per-question reads against the cached prefix.</em></p>

For technical readers: the released models use a truncated Qwen3 base model with LoRA on the upper retained layers, a candidate-aware contextual readout over the terminal decision state, and separate Choice, Noul and Score heads sharing that backbone.

We train on four kinds of decision: evidence (NLI-style), knowledge (multiple choice), workflows, and uncertainty (soft targets). The workflow portion includes counterfactual rubric groups, described below.

**A known defect in these two checkpoints.** Both were trained with a 1,024-token state window on a long-policy corpus that rendered its supporting facts at the *end* of each row. Training truncates from the right, so 98.8% of those rows lost their facts before the model saw them, and the model was optimised to answer long policies from states that no longer contained the evidence. We found this after the training runs that produced Small and Medium. It is the main reason both are weak on long-document policy (.21 for Medium), and it is why the released models are for states up to roughly 1k tokens rather than the 4k the inference package will accept. A matched 2×2 puts the cost of that defect at 11.2 points on a held-out long-state set (p = 5e-10); the corpus is regenerated facts-first and the next checkpoints train on the fixed version. The [deep dive](./technical-deep-dive.md) has the measurement.

## The released models

| model | size | JevBench standard\* | JevBench hard\* | CLINC-150 | warm p50 |
|---|---|---|---|---|---|
| `typical-small` | 1.7B | .694 | .432 | .801 | 15.5–17 ms |
| `typical-medium` | 4B | .806 | .423 | .847 | 19–21 ms |

\* Public-subset run against JevBench v1.2.1 (72 standard / 111 hard public ids), not a ranked leaderboard entry. Majority baselines on this split are .311 standard and .336 hard, and at n_eff ≈ 36 on standard, small gaps are noise. Latency is warm p50 for a single K = 2 decision over a 256-token state, one stream, in process on one H100 through the public inference package, model load excluded — not a hosted-endpoint number and not comparable to one measured over a network.

Medium is the better model for a small latency increase, on most of what we measure: it leads Small by 5 points on CLINC-150, 11 on MMLU-Pro among-K, 10 on held-out yes/no decisions and 5 on the composition curriculum. The JevBench standard gap (.694 to .806) points the same way, though at 36 independent states that tier alone would not settle it. It is not a clean sweep — Medium is 9 points worse on TREC-fine (50 fine-grained topics), and the two are within noise of each other on the hard tier. Neither released model solves the hard compositional tier: long-policy, multi-step, temporal, unit and trade-off decisions sit far below the standard tier for both.

<p align="center"><img src="../figures/fig_latency_quality.png" alt="Warm per-decision latency against JevBench standard accuracy for typical-small and typical-medium, against the latency bands of a generative LLM call" width="640"></p>
<p align="center"><em>Warm per-decision latency against the bands a normal LLM call falls into depending on how it has to answer. Both released models sit left of the fastest of those, which is a model emitting a single letter.</em></p>

One more thing about that hard tier: our probabilities there are not usable. A uniform predictor over that tier's candidate sets scores a Brier of .66; `typical-small` scores .79 and `typical-medium` .77 — worse than guessing evenly. On the standard tier the same comparison runs the other way by a distance (.40 and .30 against .69), which is where the abstention threshold below is worth using. If your decisions look like the hard tier, take the argmax if you like it, but don't threshold on the confidence.

We also trained a 14B research candidate against a release bar written down before training (hard-tier accuracy ≥ .559 or hard-tier Brier ≤ .65, and long-document policy accuracy ≥ .35). It reached .931 on the standard tier, the best number this project has produced, and missed the bar on both counts — Brier .66, long-document policy .158. So it isn't shipping. A note for anyone writing a gate like that one: ours was a point threshold on a quantity with a ±9-point interval, which is not a decidable rule. Next time it goes on a lower confidence bound.

## Rules are part of the input

A decision model that memorises "this kind of ticket gets that label" is useless the moment your policy changes. So much of the workflow training is counterfactual: the same state and options appear under different rubrics with different correct answers, forcing the model to read the rule instead of pattern-matching the state.

It works inside the rule grammar we generate and stops at its edge. On held-out rubric-flip items, where the rule is inverted and the state unchanged, `typical-small` scores .718 and `typical-medium` .743, against .477 for the same recipe trained without that corpus. On level-7 composition, which mixes temporal, unit, expected-value and trade-off reasoning and which our generator never produces, they sit at .495 and .544 — against a .441 uniform-guess rate on that set, so barely off the floor, while the same models score .84 to .90 on held-out families, grammars and styles the generator does produce. What we generate is learned. What we don't generate is not.

## Where direct decisions still break

Every model we've trained, at every size, does worse on the hard tier than on the standard tier. This is the project's clearest open problem, and it is not a capacity problem. A frozen 14B backbone with three examples in its prompt and no training at all scores .559 on that tier, against .450 for our trained 14B candidate. Both models answered the same 111 items, so the test that applies is a paired one: **frozen minus trained is +.108, 95% CI [+.027, +.189], p = .015.** The untrained backbone wins, and we can't wave it off as noise.

<p align="center"><img src="../figures/fig_hard_families.png" alt="JevBench hard-tier accuracy by family for typical-small, typical-medium and the frozen 14B baseline" width="640"></p>
<p align="center"><em>Hard-tier accuracy by family: both released models, the unreleased 14B, and the frozen 14B baseline. Long-document policy and serial-symbolic composition drag the average down independently, and the frozen backbone leads on both.</em></p>

Family by family, the frozen backbone leads our trained one on trade-off (.50 to .17), long-document policy (.42 to .16), ambiguity (.71 to .43) and temporal arithmetic (.33 to .20). We had been explaining those failures as missing training data — our generators never produce that kind of composition, so the model never learned it. That explanation predicts the frozen model fails them too, and it doesn't. So the question we can actually ask is narrower, and more awkward: what is our fine-tuning removing? The [deep dive](./technical-deep-dive.md) has the full table and the two controls that would settle it.

Two other places Typical is the wrong tool today: candidate sets in the hundreds or thousands need a retrieval front end feeding a shortlist to the decision head, and Choice keeps some sensitivity to the order you list options in. `noul()` does not, because it renders no candidates at all.

## Four things that surprised us

1. The last layer of the language model was the wrong layer to decide from.
2. Candidates had to participate in the computation, not be scored against a state computed before they arrived.
3. Abstention had to be architecture, not vocabulary.
4. One KV-cache deep copy was costing a quarter to a third of serving latency.

Each of those started as a bug or a failed run. [Read the technical deep dive →](./technical-deep-dive.md)

## What's open today

Open weights and inference code, today, on Hugging Face: [`OzLabs/typical-small`](https://huggingface.co/OzLabs/typical-small) and [`OzLabs/typical-medium`](https://huggingface.co/OzLabs/typical-medium), plus the earlier [`OzLabs/typical-small-preview`](https://huggingface.co/OzLabs/typical-small-preview) as a reference point. Each repo carries the weights, the self-contained `inference/` package, and the evaluation artefacts every number here is read from (`eval_wf_full.json`, `jevbench_summary.json`), so you can check the tables against the files rather than against us.

Both checkpoints are Apache-2.0 over Apache-2.0 Qwen3 base models, and the training recipe is documented down to the exact flags in the model cards. Training code and the full experimental report are not public yet, and neither is a packaged `pip` install. Those are what we're preparing next.

**Data licensing, stated plainly rather than buried.** The weights are Apache-2.0 and the backbones are Apache-2.0, but the training mix is not uniformly permissive and we are not going to imply it is. Two trained sources carry non-commercial licenses: ANLI is CC BY-NC 4.0 and SciQ is CC BY-NC 3.0. Two more portions of the uncertainty corpus (`metaeval/ambient`, trained on; `metaeval/chaos-mnli-ambiguity`, evaluation-only) declare no license at all on their Hugging Face cards. Whether non-commercially-licensed training data constrains the use of the resulting weights is genuinely unsettled, and we are not going to pretend to resolve it in a launch post. What we can do is tell you exactly what went in, which is the whole per-source table in each model card, so that your own counsel has something to read. It is also part of why the data release trails the weights.

## Try it

The inference package ships inside each model repo:

```bash
huggingface-cli download OzLabs/typical-small --local-dir typical-small
pip install -r typical-small/inference/requirements.txt
```

```python
import sys; sys.path.insert(0, "typical-small/inference")
from typical import Typical

m = Typical.from_pretrained("OzLabs/typical-small", device="auto")  # or "OzLabs/typical-medium"

m.choice(state, "What does the customer want?", ["refund", "replacement", "repair"])
m.noul(state,   "Is the order still under warranty?")
m.score(state,  "How urgent is this ticket?", ["0", "1", "2", "3"])
```

It needs `torch`, `transformers`, `safetensors`, `huggingface_hub` and `numpy`, and nothing from our training stack. `example.py` in the same directory runs end to end.

Pick one bounded decision your software currently makes by calling an LLM and parsing the answer. Write down the label set as it actually varies at runtime. Swap the generation step for `choice`, `noul` or `score`, and test it on your own labels. If the model abstains a lot, that's telling you something about your label set.

And run the control we keep running on ourselves: a frozen larger model with a few examples in its prompt. If it beats these, believe that number before you believe ours. We already know of cases where it does — a frozen Qwen3.5-9B reading letter logits scores .931 standard and .595 hard under a rendering we replicated from another published system, above anything we have trained at any size. That's in the deep dive too. What these buy you over it is the interface, the latency and the runtime label space. Not capability.

## Category context

TypeSafe's Jev established this class of model: state plus a typed question in, probabilities out, no text to parse. It's closed and API-only, with no published weights, and the Choice/Noul/Score vocabulary is theirs. Encoding a state once into a KV cache and reading answers off the logits isn't our idea either — it's the standard move in this category, and several open entries do it. A growing open ecosystem has formed around the same primitive, with more than forty entries on the public JevBench leaderboard.

What's ours is the rest of the stack being inspectable, and a set of published negatives: the candidate-blind architecture that failed its own control, the rendered-∅ pathology, the frozen baseline we can't beat on the hard tier. Two things in the design we'd defend as contributions rather than reimplementation are the factored abstention head and the Bernoulli Noul head, both of which fix failure modes the rendered-candidate approach has by construction.

Where we actually stand: we are not a ranked leaderboard entry, we have run the public subset only, and several entries on that board are ahead of us on the hard tier. Our training code is not public yet, which is a weaker position on inspectability than at least one competitor already holds. Those are the two things to fix next, and neither is a research problem.

Next: a calibration objective aimed at the hard tier, generator work on the composition families, a Qwen3.5 port already in progress, training-code release, and `typical-large` if the 14B clears its own bar.
