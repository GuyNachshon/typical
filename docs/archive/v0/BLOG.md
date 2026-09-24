> **Archived — typical-v0.** Describes the v0 model (run `joint_emb_lw_v5`), not the shipped
> `typical-small` (`ts1b`) / `typical-medium` (`tm1b`) checkpoints. Every number below is
> superseded by `releases/typical-small.md`, `releases/typical-medium.md` and
> `releases/typical-small-preview.md`. Kept for the writing, not the measurements.

<!--
Title options (category reframe, not a SOTA claim):
1. "Typical: a decision model, not a chat model"
2. "Read once, decide hundreds of times: Typical, an open decision model"   <- picked
3. "What if the model's output type were a probability distribution? Introducing Typical"
-->

# Read once, decide hundreds of times: Typical, an open decision model

*A 1.7B open-weights model that takes a piece of text and a list of typed questions, and returns calibrated probabilities over each question's options plus a learned "none of the above". It never generates a token.*

## TL;DR

- Typical answers `P(option | state, question, options ∪ {none})` directly. One forward pass over the state, then every question is an isolated suffix over the cached state. No generation, no JSON parsing.
- Per-decision cost barely moves with the number of options: 1.29 ms at K=4, 2.44 ms at K=1000 (256 questions over one state, A100). A query-batched log-prob baseline on the same backbone costs 4.3 ms and 651 ms.
- Calibrated where it was trained: SNLI ECE .009 raw, MNLI .026; the built-in null detects absent answers with AUROC .96–.98. Reordering or duplicating options changes nothing, exactly.
- It is not a knowledge model (MMLU-Pro .09), the state is capped at 256 tokens, and it is weaker on label vocabularies it never saw. Weights, data recipe, and training code are open; a retrain costs about $4 of A100 time.

## The problem

If you want a decision out of an LLM today, you generate text, parse it, and hope. The prompt has to restate the document for every question, so a 40-question checklist over one ticket pays for the ticket 40 times (or you cram the questions into one prompt and pray the JSON comes back well-formed). The probabilities you can extract from token logits are not the probabilities you want: the prompted Qwen3-8B baseline in our evals has an expected calibration error of .49 on SNLI and needs a temperature of about 4.6 to fix it (`REPORT.md` §3b). And there is no principled "none of the above". You can add the string "none" as an option, but a base model has no training for it: the prompted 8B picks it on 96% of out-of-scope inputs and still separates absent from present answers at only AUROC .62 (`PROJECT.md` §1, `REPORT.md` §3b).

TypeSafe's Jev, launched last week, made the case that this is a class of model worth building. We agree. Ours is small, open, and different in a few places we think matter. This post is about what it does, what it measures, and what it does not do.

## What Typical does

The model computes

    P(y | x, q, A ∪ {∅})

where `x` is an unstructured state (a ticket, a log line), `q` is a natural-language question, `A` is a candidate set defined at call time, and `∅` is "none of these" (`idea.md` §1). For M questions over one state, `x` is encoded once and each `(q_i, A_i)` runs as its own suffix over the cached state, isolated from the other questions. The output is a distribution over `A ∪ {∅}`, so the answer physically cannot be "maybe ask Sarah" (`idea.md` §2E).

```python
from typical import Typical

m = Typical.load("runs/joint_emb_lw_v5")          # or Typical.from_pretrained("<hf-repo>")
ticket = "My package arrived damaged and I want a refund."
probs = m.decide(ticket, [("What does the customer want?", ["refund", "cancel", "track package"]),
                         ("Is the customer angry?", ["yes", "no"])])
# probs[0] -> [0.965, 0.0, 0.0, 0.035]   last entry is ∅ = none of the above
```

`decide()` returns raw lists; `typical.choice`, `typical.noul` and `typical.score` turn one list into a typed answer (argmax, P(yes), probability-weighted level). All three are the same primitive with a fixed option list (`typical/__init__.py`).

## Hero figure: per-decision cost vs K

![per-decision cost vs K](site/data/bench.json → figure)

Marginal milliseconds per decision at M=256 questions over one state, A100 80GB PCIe, bf16, single stream, candidate vectors warm (`runs/bench_a100/bench.json`, `REPORT.md` §3i). The baseline is log-prob scoring on the same Qwen3-1.7B with the state prefix shared and queries batched 32 at a time; it matches the sequential version to 2e-5.

| K (options) | Typical | log-prob, query-batched | ratio |
|---|---|---|---|
| 4 | 1.29 ms | 4.3 ms | 3.3× |
| 32 | 1.27 ms | 20.3 ms | 16× |
| 150 | 1.34 ms | 92.1 ms | 69× |
| 1000 | 2.44 ms | 650.8 ms | 267× |
| **M=1, K=4 (single question, total)** | **131 ms** | **89 ms** (plain, unbatched log-prob call: **43 ms**) | **0.33–0.68×** |

The last row is the one to read first if you have one question per document. At M=1 we are three times slower than a plain prompted call, because we still pay the state encode plus a suffix pass plus an embedding call for one answer. Break-even is around M=8 against the plain baseline (138 vs 341 ms total) and around M=64 against the query-batched one (149 vs 319 ms). The curve is not flat either: the cost is an LM term independent of K plus a small term linear in candidate tokens, which is why K=1000 costs 2.4 ms rather than 1.3. Cold candidates (label strings the embedder has never seen) add 35–50% (`runs/bench_cold_emb/bench.json`). The bench ran on `joint_emb_lw`, which has the same architecture and compute as the shipped checkpoint.

## How it works

**Backbone.** Qwen3-1.7B-Base, cut at layer 20 of 28, with LoRA (r=16) on layers 13–20 (`runs/joint_emb_lw_v5/results.json` → `args`). Why not the last layer: our first version tapped layer 28 and scored 58.6 on SNLI, which is the hypothesis-only baseline; the head was reading the last layer's next-token prediction and ignoring the premise. Moving the tap to layer 20 gave 65.3 frozen and 68.2 with LoRA, and the loss finally descended (`REPORT.md` §2.4). The full-depth variant, re-run on the final recipe, lost 7 points on MNLI and 13 on BoolQ (`REPORT.md` §3g).

**State once, queries as suffixes.** The state is a prefix; its KV cache is computed once. Each question is a causal suffix over that cache, so the question attends to the state but not to the other questions. This is bit-exact with running the concatenation `state + question` from scratch (`tests/test_pipeline.py::test_joint_causal_invariance`). We went this route after noticing that the "expensive cross-encoder" we were comparing against is itself late-interaction: a causal decoder over `premise [SEP] hypothesis` never lets the premise see the hypothesis (`REPORT.md` §2.8). Keeping pretrained layers over a cached prefix is the standard trick from DeFormer, PreTTR and Poly-encoders; we just apply it to decisions.

**Candidates.** Each option string is embedded by Qwen3-Embedding-0.6B (frozen). We tried mean-pooled decoder features first; switching to the embedder was the single best change we made, lifting paraphrased-label accuracy from .55 to .81 and Banking77 (a label set never seen in training) from .30 to .44 (`REPORT.md` §3c). Candidate vectors are cached, so a fixed label vocabulary costs nothing per call.

**Scorer and null.** An energy scorer pairs the query representation with each candidate; the null `∅` is a learned score that sees the candidate set, not just the state. The state-only version leaked its training base rate: it learned that 13% of rows had no answer and put that prior on every clean item (`REPORT.md` §2.3).

**Listwise mixer.** The shipped checkpoint has a small SetMixer over the candidate set before scoring (identity at init). What it buys: out-of-scope recall +17 to +22 points and half the K-dilution of the null. What it costs: 4–7 points on the one large-K in-distribution set (CLINC-150) and, by design, exact add-irrelevant invariance (`REPORT.md` §3d(2)). We also ship the pointwise checkpoint for anyone who wants that invariance back.

**Temperature.** One scalar T fitted on validation (T=1.22 for this checkpoint). We report ECE before and after it, because the "after" number is selected on val.

## Results

All numbers are `joint_emb_lw_v5`, seed 0, temperature-scaled unless marked raw (`runs/joint_emb_lw_v5/results.json`, `REPORT.md` §3e). Seed floor from two seed pairs: NLI ±0.7, trained large-K ±3, unseen label spaces and OOS ±5 (`REPORT.md` §3d(3)). Differences under those are ties.

**Evidence decisions** (state = premise/passage/utterance; options = label strings at runtime):

| | Typical | fine-tuned cross-encoder `C_lora` | prompted Qwen3-8B log-prob | published encoders (fixed heads, full fine-tune) |
|---|---|---|---|---|
| SNLI | .909 | .855 | .818 | – |
| MNLI-m | .880 | .768 | .823 | RoBERTa-large 90.2, DeBERTa-V3-large 91.8 |
| ANLI (R1–3) | .555 | .376 | .504 | RoBERTa-large ≈ 54.9 |
| BoolQ | .834 | .598 | .863 | RoBERTa-large 86.9 |
| CLINC-150 (151-way incl. ∅) | .784 | .393 | .387 | BERT 96.9 |
| HWU64 | .818 | .623 | .020 | BERT 92.1 |

Baselines from `REPORT.md` §3b; references from `COMPARE.md` §2a. Caveats: BoolQ and HWU64 are inflated by at most 1–2 points by near-duplicates between train and eval (`runs/leak_audit.json`). The published encoders train a fixed classifier per task; ours reads the label strings at call time and has to beat a null. On CLINC-150 a supervised BERT is 19 points better than us; we do not claim otherwise.

**Calibration** (in-distribution; ECE raw → scaled): SNLI .009 → .020, MNLI .026 → .011, BoolQ .057 → .034, CLINC .065 → .034. The SNLI number got *worse* after scaling because the global T is fitted on a classification-heavy val set and over-softens NLI (`REPORT.md` §3e). Out of distribution it is not calibrated: Banking77 ECE .21 raw.

**Null.** AUROC for "the answer is not in the set": .961 on CLINC with random gold removal, .980 on SNLI, .975 on SQuAD-style unanswerable. CLINC out-of-scope recall .735. The honest curve: P(∅ | gold absent) falls from .94 at K=2 to .41 at K=150 (`ksweep_clinc`). About half of that is difficulty (at K=150 the removed gold's sibling intents are still in the set) and half is real K-dependence, an extreme-value effect over 149 distractors; excluding siblings gives .95 → .59 (`REPORT.md` §3h). Neither a factored null nor a K-aware bias fixed it; we tried both.

**Choice-set behaviour** (`cse_clinc`, `cse_banking77` in `results.json`). Reordering options: max Δp 0.000. Duplicating an option: the two copies get identical scores (slot gap 0.0); mass error .015. Adding an irrelevant option: the listwise checkpoint shifts the top-2 log-odds by .20 (CLINC) / .24 (Banking77); the pointwise checkpoint shifts them by 0.000, exactly, because its scorer never looks at the other candidates. For context, a third-party probe of Jev measured −0.28 log-odds from one added irrelevant option and reorder accuracy moving 0.84 → 0.93 (https://archerhume.com/posts/jevs-architecture-unmasked); different protocol, different items, so read it as context, not a head-to-head.

## Things we got wrong on the way

We think these are more useful than the table above, so here they are in the order they bit us (`REPORT.md` §2).

1. **Attention sink.** Qwen3 has no BOS token, so position 0's hidden state is content-independent (cosine 0.9998 between "neutral", "yes" and "transfer"). Every one-token candidate was the same vector. Fix: prepend `<|endoftext|>` and drop it.
2. **A fixed classifier in disguise.** With three fixed NLI label strings in training, the scorer memorised three vectors; paraphrased labels scored 11%, below chance. Fix: randomise label wordings in training, hold five wordings out for eval.
3. **State-only null leaked its base rate.** Described above; the fix was making the null see the candidate set.
4. **Last-layer tap.** The big one: the last layer of a base LM is a next-token predictor, and for NLI that means a hypothesis-only model. Backbone size was never the bottleneck; 0.6B → 1.7B bought nothing until the tap moved.
5. **Rogue dimensions.** Three hidden dimensions carried ~30% of token norm. Per-dimension z-scoring was worth +3.5 on SNLI and +2–4 on intents.
6. **Null over-fire from a data bug.** Our synthetic "answer absent" rows used near-miss sets of size N−1, which taught "almost the whole label set ⇒ none", at a 25% rate. Halving the rate and using K ∈ {3, 10} doubled TREC and 20NG accuracy.

Two more from the eval side: an early "73× cheaper at K=4" was against a baseline that did not batch across queries; the fair number is 3.3× (`REPORT.md` §3i). And the K-sweep target we pre-registered (null range ≤ .10) turned out to be mis-specified, because the sweep conflates K with difficulty (`REPORT.md` §3h).

## What it is not

Typical is not a knowledge model and not an LLM replacement: it decides from the text you give it (≤250 tokens), scores 9% on MMLU-Pro, generates nothing, and is less accurate on label vocabularies it never saw ([LIMITATIONS.md](LIMITATIONS.md)).

The MMLU-Pro number deserves a sentence. Nothing in training is parametric-knowledge QA; every task is "the answer is in the state", and we cut the backbone at layer 20, before the layers where a base LM forms answers. The same backbone with a letter-readout head, trained on the same data, gets .235, because reading the options against each other in context is what that benchmark rewards (`REPORT.md` §3f). Jev reports 84.6% there. Different class of model, not a worse version of the same one.

Other limits: the state is truncated at 256 tokens, questions at 64, candidates at 32 embedding tokens (`typical/__init__.py`); we have never benchmarked past that. English only. One seed. Banking77, a label set never seen in training, is .527 with 77 options; a supervised model is in the 90s. The Score primitive is derived (probability-weighted level over an ordered option list), not trained. Calibration is only measured in-distribution.

## Where it sits vs Jev

Jev and Typical share the same shape: shared state, isolated question branches, direct probability readouts, three primitives (`COMPARE.md` §1). Jev is ~10B active (inferred), closed, reads option text in context and has no built-in null; Typical is 1.7B, open, scores cached candidate embeddings and has a learned null. On knowledge (MMLU-Pro) Jev is far ahead; on choice-set invariance and null detection we measure things Jev's docs do not report. It is not apples-to-apples: their accuracy numbers are a vendor workflow suite with LLM-generated labels, their latency is an API round-trip on unknown hardware, and ours are public NLU sets on a rented GPU (`COMPARE.md` §3). What would settle it: Jev's API on our public eval files with our label strings plus a literal "none of the above", scored with the same `metrics.py`; the run costs about $0.30 in API calls and we will publish it as soon as we have access.

## Use it

Serve the model and the demo site locally:

```
uv run python -m typical.serve --run runs/joint_emb_lw_v5 --pointwise runs/joint_emb --port 8000
```

The same request body TypeSafe's SDK sends works unchanged against `POST /v1/systemone`; answers carry an extra `p_none`:

```
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "My package arrived damaged and I want a refund.",
  "model": "typical-v0",
  "questions": {
    "department": {"type": "choice",
                   "instructions": "Which department should handle this ticket?",
                   "criteria": {"returns": "", "shipping": "", "billing": ""}},
    "urgent": {"type": "noul", "instructions": "Does this need a reply today?"}
  }}'
```

Weights: `<HF_REPO_PLACEHOLDER>` (`typical-v0`, listwise) and `<HF_REPO_PLACEHOLDER>/pointwise` (`typical-v0-pointwise`, exact add-irrelevant invariance). Load either with `Typical.from_pretrained(repo)`.

Retrain from scratch (12k steps, batch 64, one A100, about $4 of rented time; `PROJECT.md` §3):

```
uv run data.py --out data_v5
uv run python train.py --name typical-v0 --joint --tower_layers 0 --tap_layer 20 --zscore --lora_r 16 \
    --data data_v5 --cand_encoder qwen3emb --listwise --steps 12000 --bs 64 --seed 0
```

## What's next

- A second seed of the shipped checkpoint, so every number above gets an error bar instead of a floor borrowed from sibling runs.
- A data-licence decision: ANLI is CC BY-NC 4.0, which means either non-commercial weights or dropping ANLI and retraining (~$5). The model card lists every one of the ~30 training sources with its licence.
- A native-choice router. Our research branch `nc_n3` reads options in context and reaches MMLU-Pro .314 among-K without generation, but has no shared-state inference path and a null that abstains on 61% of items (`REPORT.md` §3l). Routing knowledge-style questions there and evidence questions here is the obvious next model; it is not in this release.
- Longer states. The 256-token cap is a training-data limit, not an architectural one; the attention cost per query against a long prefix is unmeasured.
- A trained Score primitive, rather than the derived one.

## Links

- Code, training recipe, eval harness: this repository (`train.py`, `data.py`, `metrics.py`, `bench.py`)
- Weights: `<HF_REPO_PLACEHOLDER>`
- Full results: `runs/joint_emb_lw_v5/results.json`, `runs/joint_emb/results.json`, `runs/bench_a100/bench.json`
- Long-form report with every failed experiment: `REPORT.md`; comparison notes: `COMPARE.md`; limitations: `LIMITATIONS.md`
- Third-party analysis of Jev we cite for context: https://archerhume.com/posts/jevs-architecture-unmasked
