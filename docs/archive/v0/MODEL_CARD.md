---
license: other
license_name: pending-data-audit
license_link: LICENSE
base_model: Qwen/Qwen3-1.7B-Base
library_name: typical
pipeline_tag: text-classification
language:
  - en
tags:
  - decision-model
  - calibrated-probabilities
  - abstention
  - zero-shot-classification
  - natural-language-inference
  - intent-classification
  - lora
  - qwen3
datasets:
  - stanfordnlp/snli
  - nyu-mll/multi_nli
  - facebook/anli
  - Zhengping/UNLI
  - google/boolq
  - rajpurkar/squad_v2
  - clinc/clinc_oos
  - fancyzhx/ag_news
  - fancyzhx/dbpedia_14
  - community-datasets/yahoo_answers_topics
  - cardiffnlp/tweet_eval
  - dair-ai/emotion
  - SetFit/sst5
  - FastFit/hwu_64
  - benayas/snips
  - mteb/amazon_massive_intent
  - mteb/mtop_intent
  - SetFit/bbc-news
  - SetFit/subj
  - SetFit/CR
  - SetFit/enron_spam
  - SetFit/amazon_counterfactual_en
  - SetFit/sst2
  - ccdv/arxiv-classification
  - ccdv/patent-classification
  - bitext/Bitext-customer-support-llm-chatbot-training-dataset
  - SetFit/student-question-categories
  - SetFit/hate_speech_offensive
  - stanfordnlp/imdb
  - SetFit/yelp_review_full
  - google-research-datasets/go_emotions
---

> **Archived — typical-v0.** Describes the v0 model (run `joint_emb_lw_v5`), not the shipped
> `typical-small` (`ts1b`) / `typical-medium` (`tm1b`) checkpoints. Every number below is
> superseded by `releases/typical-small.md`, `releases/typical-medium.md` and
> `releases/typical-small-preview.md`. Kept for the writing, not the measurements.

# Typical v0 (`guychuk/typical-v0`)

## What it is

Typical is an open decision model: it reads a short text state once and answers many typed questions over it in parallel, each as a calibrated probability distribution over caller-supplied options plus an explicit "none of the above" (∅). It does not generate text; a question costs about 1.3 ms on an A100 once the state is encoded, roughly independent of how many options it has. It is a 1.7B-parameter Qwen3 backbone with a LoRA and a small scoring head, trained for ≈ $4–5 of GPU time on public evidence-grounded datasets (NLI, reading comprehension, intent and topic classification).

Internal name: PCDM (`runs/joint_emb_lw_v5`). Code: Apache-2.0. Weights: see License.

## Intended use

- Decisions whose answer is supported by the text you pass in: entailment/contradiction, yes/no questions about a passage, routing an utterance to one of K intents, topic or sentiment labels, "which of these is mentioned", with abstention when none applies.
- Many questions over one state (support ticket + 20 questions, log line + 150 labels), where latency and cost per decision matter.
- Local / self-hosted use; the server speaks a Jev-compatible wire format (`POST /v1/systemone`) so existing clients can be pointed at it.

## Not intended use

- Anything that needs knowledge that is not in the state. MMLU-Pro accuracy is .092 (among-K .123), chance is .10 (`runs/mmlu_joint_emb_lw_v5/results.json`, REPORT.md §3f).
- States longer than 256 tokens (truncated), non-English text, JSON/structured states (untested).
- Label vocabularies far from the training ones without checking accuracy first (Banking77-77 unseen: .527 vs .937 supervised BERT).
- Safety-critical decisions. No adversarial robustness testing was done. See [LIMITATIONS.md](LIMITATIONS.md).

## Quick start

Python:

```python
from typical import Typical
m = Typical.from_pretrained("guychuk/typical-v0")
state = "My package arrived damaged and I want a refund."
probs = m.decide(state, [("What does the customer want?", ["refund", "cancel", "track package"]),
                         ("Is the customer angry?", ["yes", "no"])])
print(probs)  # per question: [P(opt_1), ..., P(opt_K), P(∅)]
```

Server (serves the demo site and two JSON endpoints):

```bash
uv run python -m typical.serve --run runs/joint_emb_lw_v5 --pointwise runs/joint_emb --port 8000
```

Jev-wire request (same body the TypeSafe SDK sends; answers carry an extra `p_none`):

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "My package arrived damaged and I want a refund.",
  "model": "jev-latest",
  "questions": {
    "department": {"type": "choice", "instructions": "Which department should handle this ticket?",
                   "criteria": {"returns": "refunds", "shipping": "delivery", "billing": "charges"}},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "frustration": {"type": "score", "criteria": ["calm", "annoyed", "angry"]}
  }}'
```

## Interface and limits

| | value | source |
|---|---|---|
| state | ≤ 256 backbone tokens, truncated silently | `typical/__init__.py` `STATE_TOKENS` |
| question | ≤ 64 tokens | `QUERY_TOKENS` |
| each option | ≤ 32 embedder tokens; label strings only, option descriptions are ignored by the wire adapter | `CAND_TOKENS`; `typical/serve.py` |
| options per question | ≥ 1; no hard cap (benchmarked to K = 1000) | `runs/bench_a100/bench.json` |
| output | K option probabilities + one trailing P(∅); the K + 1 entries sum to 1 | `Typical.decide` |
| ∅ semantics | a learned, candidate-aware logit in the same softmax; trained on gold-absent, irrelevant-question and near-miss rows. Wire helpers renormalise the K options to sum 1 and report `p_none` separately | `typical/__init__.py` `_renorm` |
| temperature | `decide(temperature=None)` applies the val-fitted T = 1.215 (`results.json["T"]`); `temperature=1.0` returns the raw softmax | `results.json` |
| confidence | derived, not learned: `(p_max − 1/K)/(1 − 1/K)` over the renormalised options (same convention as TypeSafe's open adapter) | `typical.confidence` |
| Score / yes-no | derived from Choice with a fixed option list; nothing in the model is specific to them | `typical.score`, `typical.noul` |

## Architecture

- Backbone: `Qwen/Qwen3-1.7B-Base`, truncated at layer 20 of 28; per-dimension z-scored features (`--tap_layer 20 --zscore`).
- LoRA r = 16 on layers 13–20 (hand-rolled `LoRALinear`, not PEFT); layers 1–12 frozen. Checkpoint `best.pt` = LoRA + head, ≈ 50 MB.
- State encoded once as a causal prefix (KV cache); every question is an isolated suffix over it (no question↔question attention; bit-exact with concatenation, `tests/test_pipeline.py::test_joint_causal_invariance`). No cross-attention tower (`--tower_layers 0`).
- Candidates: option strings embedded by frozen `Qwen/Qwen3-Embedding-0.6B` (1024-d), cached per string; K-independent per suffix.
- Scorer: energy head over (query state, candidate vector) plus hybrid frozen-space similarity features; ≈ 13.5M parameters.
- Listwise `SetMixer` block (≈ 2.1M) over the candidate set (`--listwise`). This is what makes the null better and add-irrelevant IIA non-exact; see Choice-set behaviour.
- Null: learned candidate-aware ∅ logit inside the softmax (`--null softmax`, the default).
- Loss: soft cross-entropy over `[(1 − p∅)·target, p∅]`; 12,000 steps × 64, AdamW, one global temperature fitted on val.

## Training data

Rows = cap on raw source rows in `data.py` (v5 build: `--cls_cap 4000` on every classification source, CLINC untouched); each row is expanded with query templates and label-wording augmentation, so the final row count is larger (REPORT.md §1 reports 758k rows for v4; the v5 total is in the build log, not in this repo). Licenses marked "check" were not verified for this card; treat them as unresolved.

| source | HF id | family | raw rows (cap) | license |
|---|---|---|---|---|
| SNLI | `stanfordnlp/snli` (+ 5-vote `snli_1.0.zip` for soft labels) | NLI | 150,000 (+10,000 soft) | CC BY-SA 4.0 |
| MultiNLI | `nyu-mll/multi_nli` | NLI | 120,000 | mixed (CC BY-SA 3.0 / OANC / other) — check |
| ANLI R1–R3 | `facebook/anli` | NLI | R1 + R2 + 40,000 of R3 | **CC BY-NC 4.0 — non-commercial** |
| UNLI | `Zhengping/UNLI` | soft proposition | 55,000 | check (built on SNLI) |
| BoolQ | `google/boolq` | yes/no QA | all train (9,427) | CC BY-SA 3.0 |
| SQuAD v2 | `rajpurkar/squad_v2` | extractive QA with real nulls | 60,000 answerable + 40,000 unanswerable | CC BY-SA 4.0 |
| CLINC-150 (plus) | `clinc/clinc_oos` | intent + OOS | 8,000 in-scope + all OOS; 30 intents held out | CC BY 3.0 |
| AG News | `fancyzhx/ag_news` | topic | ≤ 4,000 | check |
| DBpedia-14 | `fancyzhx/dbpedia_14` | topic | ≤ 4,000 | CC BY-SA 3.0 |
| Yahoo Answers topics | `community-datasets/yahoo_answers_topics` | topic | ≤ 4,000 | check |
| TweetEval (emotion, sentiment, hate, irony, offensive, stance-abortion) | `cardiffnlp/tweet_eval` | emotion/stance | ≤ 4,000 each | check (Twitter TOS) |
| Emotion | `dair-ai/emotion` | emotion | ≤ 4,000 | other — research/educational use only |
| SST-5 / SST-2 | `SetFit/sst5`, `SetFit/sst2` | sentiment | ≤ 4,000 each | check |
| HWU64 | `FastFit/hwu_64` | intent | ≤ 4,000 | CC BY 4.0 — check |
| SNIPS | `benayas/snips` | intent | ≤ 4,000 | check |
| MASSIVE (en) | `mteb/amazon_massive_intent` | intent | ≤ 4,000 | CC BY 4.0 |
| MTOP (en) | `mteb/mtop_intent` | intent | ≤ 4,000 | CC BY-SA 4.0 — check |
| BBC News | `SetFit/bbc-news` | topic | ≤ 4,000 | check |
| Subjectivity | `SetFit/subj` | sentiment | ≤ 4,000 | check |
| Customer Reviews | `SetFit/CR` | sentiment | ≤ 4,000 | check |
| Enron spam | `SetFit/enron_spam` | topic | ≤ 4,000 | check |
| Amazon counterfactual (en) | `SetFit/amazon_counterfactual_en` | classification | ≤ 4,000 | CC BY-SA 4.0 — check |
| arXiv-11 | `ccdv/arxiv-classification` | topic | ≤ 4,000 | check |
| Patent-9 | `ccdv/patent-classification` | topic | ≤ 4,000 | check |
| Bitext customer support (27 intents) | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | intent | ≤ 4,000 | CDLA-Sharing-1.0 — check |
| Student question categories | `SetFit/student-question-categories` | topic | ≤ 4,000 | check |
| Hate speech / offensive | `SetFit/hate_speech_offensive` | classification | ≤ 4,000 | check |
| IMDB | `stanfordnlp/imdb` | sentiment | ≤ 4,000 | other — Stanford terms, restricted |
| Yelp-5 | `SetFit/yelp_review_full` | sentiment | ≤ 4,000 | other — Yelp dataset terms, restricted |
| GoEmotions | `google-research-datasets/go_emotions` | emotion | ≤ 4,000 (single-label rows) | Apache-2.0 |

Held out entirely (eval only): `mteb/banking77` (CC BY 4.0), `SetFit/TREC-QC` (check), `SetFit/20_newsgroups` (check), `metaeval/chaos-mnli-ambiguity` (check), 30 CLINC intents (`held_out_intents.json`), `TIGER-Lab/MMLU-Pro` 1,200-item slice (`frozen/mmlu_pro.jsonl`; MIT). Near-duplicate audit train vs eval: `runs/leak_audit.json` (MinHash char-5-gram); BoolQ 3.9 % and HWU64 2.6 % of eval states have a near-duplicate in train.

## Evaluation

All numbers: `runs/joint_emb_lw_v5/results.json`, seed 0, single run. "raw" = softmax at T = 1; "scaled" = T = 1.215 fitted on the v5 val set. acc counts ∅ as wrong when the gold is present; among-K (`acc_k`) scores only the K options. Sets are capped at ~3–5k items (`runs/leak_audit.json` has per-set n).

| set | acc | among-K | ECE raw | ECE scaled | NLL raw | NLL scaled | null AUROC |
|---|---|---|---|---|---|---|---|
| SNLI test | .909 | .909 | .009 | .020 | .254 | .257 | – |
| MNLI val-m | .880 | .880 | .026 | .011 | .334 | .325 | – |
| ANLI test (R1–3) | .555 | .555 | .209 | .170 | 1.113 | 1.022 | – |
| BoolQ val | .834 | .835 | .057 | .034 | .426 | .404 | – |
| CLINC-150 test (in-scope, 151-way) | .784 | .828 | .065 | .034 | .899 | .830 | – |
| CLINC-OOS (null recall) | .735 | – | .074 | .079 | .809 | .815 | – |
| CLINC-K (gold absent half) | .875 | .954 | .049 | .029 | .400 | .374 | .961 |
| HWU64 test | .818 | .837 | .040 | .041 | .657 | .638 | – |
| SNLI-null | .886 | .949 | .020 | .016 | .347 | .338 | .980 |
| SQuAD-null | .887 | .907 | .034 | .019 | .352 | .332 | .975 |
| irrelevant-question (intent / squad) | .907 / .908 | .968 / .917 | .021 / .021 | .039 / .021 | .285 / .268 | .292 / .269 | .972 / .984 |
| near-miss null (clinc / hwu64 / banking77) | .753 / .763 / .543 | .927 / .932 / .774 | .141 / .111 / .292 | .115 / .085 / .255 | .771 / .670 / 1.763 | .690 / .623 / 1.539 | .906 / .942 / .772 |
| SNLI paraphrased labels | .837 | .893 | .096 | .145 | .481 | .525 | – |
| SNLI hypothesis-only (artefact probe) | .410 | .410 | .398 | .360 | 1.935 | 1.681 | – |
| CLINC held-out 30 intents (unseen) | .762 | .876 | .130 | .102 | .845 | .752 | – |
| Banking77-77 (unseen vocabulary) | .527 | .542 | .213 | .157 | 2.286 | 2.037 | – |
| Banking77-K (unseen, gold-absent half) | .606 | .817 | .238 | .202 | 1.422 | 1.254 | .805 |
| TREC coarse / fine (unseen) | .356 / .308 | .422 / .310 | .264 / .178 | .205 / .100 | 2.285 / 3.140 | 2.076 / 2.914 | – |
| 20 Newsgroups (unseen) | .336 | .357 | .171 | .105 | 2.285 | 2.161 | – |
| ChaosNLI-M (NLL vs human label distribution) | – | – | – | – | 1.445 | 1.273 | – |
| UNLI (NLL) | – | – | – | – | .537 | .527 | .919 |
| MMLU-Pro 1,200 (`runs/mmlu_joint_emb_lw_v5`) | .092 | .123 | .395 | .385 | 3.944 | 3.850 | – |

Pointwise variant `typical-v0-pointwise` (`runs/joint_emb/results.json`, T = 0.962, data v4):

| set | acc | among-K | ECE raw | ECE scaled | null AUROC |
|---|---|---|---|---|---|
| SNLI / MNLI / ANLI / BoolQ | .906 / .873 / .524 / .817 | – | .007 / .015 / .195 / .043 | .005 / .016 / .204 / .049 | – |
| CLINC-150 / HWU64 | .803 / .875 | .821 / .883 | .025 / .015 | .029 / .025 | – |
| CLINC-OOS recall / CLINC-K | .649 / .879 | – / .955 | .174 / .029 | .162 / .033 | – / .953 |
| Banking77-77 / TREC-coarse / 20NG | .440 / .210 / .273 | .459 / .382 / .310 | .128 / .353 / .132 | .143 / .361 / .145 | – |
| MMLU-Pro | .082 | .132 | .33 | – | – |

Latency (`runs/bench_a100/bench.json`; A100 80 GB PCIe, bf16, checkpoint `runs/joint_emb_lw` — same architecture as v5, so the timing transfers; default ≈ 212-token state, 9-token queries, candidate vectors warm, single stream). Baseline `B_batched` = log-prob scoring on the same Qwen3-1.7B with the state prefix shared and queries batched in chunks of 32. Per-decision marginal at M = 256 questions:

| K | ours | B_batched | ratio | ours total (M = 256) | peak mem ours / B_batched |
|---|---|---|---|---|---|
| 4 | 1.29 ms | 4.33 ms | 3.3× | 355 ms | 8.8 / 13.6 GB |
| 32 | 1.27 ms | 20.3 ms | 16× | 346 ms | 8.8 / 25.9 GB |
| 150 | 1.34 ms | 92.1 ms | 69× | 362 ms | 9.1 / 25.4 GB |
| 1000 | 2.44 ms | 651 ms | 267× | 643 ms | 14.6 / 46.0 GB |

M = 1 (one question, K = 4): ours 131 ms total vs 43 ms for the plain log-prob baseline — we are slower until M ≥ 8 (ours 138 ms vs 341 ms). Cold candidates (not in the vector cache) add 35–50 % per decision (`runs/bench_cold_emb/bench.json`: 1.65 → 2.23 ms at K = 4, 2.52 → 3.76 ms at K = 150). Never benchmarked past ≈ 212 state tokens.

Seed floor (two seed pairs, REPORT.md §3d(3)): NLI ± 0.7, trained large-K sets ± 3, unseen label spaces and OOS ± 5 points. Any difference below that is noise. This model is one seed.

Near-duplicate inflation: BoolQ and HWU64 are ≤ 1–2 points inflated by eval states with near-duplicates in train (REVIEW.md §6, `runs/leak_audit.json`).

## Calibration note

The T = 1.215 was fitted on a classification-heavy val set and over-softens NLI: SNLI raw ECE .009 → .020 scaled, while MNLI improves .026 → .011 and CLINC .065 → .034 (REPORT.md §3e). Use `temperature=1.0` for NLI-style questions if ECE there matters. In-distribution ECE is ≤ .06; on unseen vocabularies it is .13–.29 raw (CLINC-heldout .130, Banking77 .213, near-miss Banking77 .292). Calibration under domain shift is not a property this model has.

## Choice-set behaviour

`cse_*` battery, `runs/joint_emb_lw_v5/results.json` (listwise) and `runs/joint_emb/results.json` (pointwise):

| effect | listwise (v0) | pointwise (v0-pointwise) |
|---|---|---|
| reorder options: max Δp | < 1e-5 (exact) | < 1e-5 (exact) |
| duplicate an option: slot gap | 0.0 (exact) | 0.0 (exact) |
| duplicate an option: mass error | .015 clinc / .030 banking77 | .015 / .047 |
| add an irrelevant option: Δ log-odds of top-2 | **.204 clinc / .243 banking77 / .205 hwu64 / .035 snli** | **0.000** (4e-8) |
| add irrelevant: P(irrelevant option) | .002 clinc | .005 clinc |
| remove the gold option: ΔP(∅) | +.695 clinc / +.324 banking77 | +.746 / +.285 |

The listwise SetMixer sees the whole option set, so adding an option can move the others (IIA is not exact, by design); it buys a better null (OOS recall .735 vs .649). The pointwise checkpoint scores each option independently and has exact add-irrelevant invariance. Pick by need.

## Known limitations

See [LIMITATIONS.md](LIMITATIONS.md). Short list: no parametric knowledge (MMLU-Pro .092); 256-token state; accuracy drops on label vocabularies it never saw (novelty is partly read as "none of the above"); P(∅ | gold absent) falls from .94 at K = 2 to .41 at K = 150; OOD calibration is poor; one seed; English only; slower than a log-prob baseline for a single question.

## Reproduce

Cost of one training run: ≈ 1.3 H100-hours ≈ $4 (COMPARE.md §2); v5 was trained on an A100 for ≈ $5 (REPORT.md §3e).

```bash
uv sync
uv run pytest tests/ -q                                    # CPU, < 60 s

# data v5 (label-diverse classification, --cls_cap 4000; eval files byte-identical to v4)
uv run hf download guychuk/pcdm-data --repo-type dataset --include 'v5/*' --local-dir .
# or rebuild: uv run data.py --out data_v5 --cls_cap 4000 && uv run data.py --selftest --out data_v5

# train (= runs/joint_emb_lw_v5; drop --listwise for the pointwise variant, --data data_v4 for joint_emb)
uv run python train.py --name joint_emb_lw_v5 --joint --tower_layers 0 --tap_layer 20 --zscore --lora_r 16 \
    --data data_v5 --cand_encoder qwen3emb --listwise --eval_every 6000 --wandb --hf_repo guychuk/pcdm-runs

# eval-only from best.pt (architecture read from the checkpoint) + per-item logits dump
uv run python train.py --name joint_emb_lw_v5 --eval_only --data data_v5 --dump_logits runs/dump_joint_emb_lw_v5

# side-by-side table of every runs/*/results.json (scaled; --raw for T = 1)
uv run report.py

# latency
uv run bench.py --model runs/joint_emb_lw_v5 --backbone Qwen/Qwen3-1.7B-Base --name bench_a100 --full
```

The `--cls_cap 4000` build flag is reconstructed from REPORT.md §3e ("30 label vocabularies × ≤ 4,000 rows") and `data.py --cls_cap`; the exact v5 build invocation is not in the repo — download the published `v5/` to be safe.

## Variants

- `typical-v0` (this repo, `runs/joint_emb_lw_v5`): listwise SetMixer, best null, default in the server and site.
- `typical-v0-pointwise` (`runs/joint_emb`, data v4): no SetMixer; exact add-irrelevant IIA (Δ log-odds 0.000), better in-distribution intent accuracy (CLINC .803, HWU64 .875), weaker null (OOS recall .649) and weaker on unseen vocabularies (Banking77-77 .440). Load with `--pointwise runs/joint_emb`.
- `nc_n3` research preview (`runs/nc_n3`, not demoed, not wrapped by `typical`): a different readout — options are rendered into the suffix and scored against their own contextual hidden states through all 28 layers, trained on data v5 plus a knowledge-MCQ corpus (`scripts/distill_corpus.py`: MMLU auxiliary-train, ARC, OpenBookQA, CommonsenseQA, SciQ, QASC, LogiQA2, AQuA-RAT, MedMCQA). It reaches MMLU-Pro among-K .314 (acc .133) and CLINC-heldout .916, but its null is broken (abstains on 18 % of MMLU-Pro items, P(∅ | absent) .99 at K = 150 while CLINC-OOS recall is .20), evidence accuracy regresses (SNLI .863, MNLI .752, BoolQ .737), IIA Δ .126, and it has no shared-state inference path, so none of the latency numbers apply. REPORT.md §3l. Published for research only.

## Citation

```bibtex
@software{typical2026,
  title  = {Typical: an open decision model},
  author = {Nachshon, Guy},
  year   = {2026},
  url    = {https://huggingface.co/guychuk/typical-v0},
  note   = {v0, checkpoint joint_emb_lw_v5}
}
```
