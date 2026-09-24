# PCDM vs TypeSafe Jev / "System One" — comparison (2026-09-18)

Sources: local `REPORT.md` §3c/§3d/§4, `REVIEW.md`, `runs/{bench_fair,joint_emb,joint_emb_lw,joint_emb_s1,mcq_lora,B_8B}/results.json`,
`uv run pcdm/report.py`; external pages fetched today (full URLs in §5). Every Jev cell carries a source tag: [TS-blog], [TS-docs-*],
[TS-evals], [AH] (third-party black-box analysis, archerhume.com, 17 Sep 2026), or **n/d** = not disclosed. "Inferred" cells say how.
"Ours" = `joint_emb` (seed 0) unless noted; `joint_emb_lw` where the null matters; latency from `bench_fair` (checkpoint `joint_v1`).

## 1. Framing

PCDM claims (IDEA2 §1–2): a pretrained LM turned into a non-generative decision engine — encode the state once, run many isolated
query suffixes against the cached prefix, score a *runtime-defined* candidate set with an energy head plus an *architectural* null,
and emit calibrated distributions. TypeSafe claims, in their words: System One models are "a new class of frontier models built
to make fast, structured decisions that software can use directly"; Jev "achieves similar levels of intelligence on System One
tasks compared to existing LLMs, while being two orders of magnitude faster and more efficient"; it "generates all outputs in a
single query", output is "type-safe structured values … All answers are accompanied with calibrated probabilities and confidence
scores", trained with "Reinforcement Learning for Calibrated Decisions (RLCD)" [TS-blog]; "Outcomes assigned a probability of 0.8
should occur about 80% of the time" [TS-docs-primer]. The black-box analysis [AH] infers "shared-state encoding, isolated
question branches, and direct probability readouts instead of text generation" — i.e. the same three structural commitments as
PCDM's joint architecture. Overlap: shared prefix + isolated branches + direct probabilities + three primitives (Choice/Score/Noul ≈
our Choice/Score/Proposition). Differences: Jev reads option text in context (listwise; IIA and order effects measured), has no
architectural null (docs say add an "other / none of the above" option), is ~10B-active (inferred) and proprietary; PCDM is
1.7B (20 layers), open, candidate embeddings cached and K-independent, learned null, permutation-equivariant, and measurably weak
on label vocabularies it never saw.

## 2. Comparison table

| axis | PCDM (ours) | Jev / System One | source (Jev) |
|---|---|---|---|
| backbone | Qwen3-1.7B-Base, truncated at layer 20, LoRA r16 on layers 13–20 (REPORT §1) | n/d. Tokenizer closest to Qwen (348/415 agreement) but "doesn't reveal one"; "every pretrained model at that scale is a causal decoder" | [AH] |
| size | 1.7B total, ~1.2B used (20/28 layers); head ≈ 13.5M + SetMixer 2.1M (lw) | n/d. Inferred "MoE with about 10B active parameters" from "about 30k tokens in roughly 160 ms"; author: "an inference, not a measurement" | [AH] |
| candidate interface | list of label strings (≤16 tokens each), embedded by frozen Qwen3-Embedding-0.6B, cached per string; scorer is K-independent per suffix | `criteria`: map option-name → description (string/object/array); "All option names and descriptions are visible to the model"; ≤255 options/question; ≤32k tokens per branch, 64k per request | [TS-docs-choice], [TS-docs-models], [AH] |
| primitives | Choice (+null); Proposition via NLI/UNLI/BoolQ families; **Score not implemented** (REPORT §6) | Choice, Score (probability-weighted level), Noul (calibrated yes/no) | [TS-docs-system-one] |
| null / abstain | learned candidate-aware ∅ logit in the softmax, trained on gold-absent / irrelevant-question / near-miss rows | none built in; docs: include "an `other` or `none of the above` option when the list might not cover every input" | [TS-docs-choice]; [AH] "no explicit discussion" |
| confidence | derived (p_max, entropy, margin); one global T fitted on val (T=0.96 for `joint_emb`, raw≈scaled) | derived, not learned: Choice `c = (p_max − 1/K)/(1 − 1/K)` (official adapter); Noul carries none; thresholds <0.5 / 0.5–0.9 / >0.9 recommended | [AH], [TS-docs-confidence] |
| state reuse | causal prefix KV cache, bit-exact vs concatenation (Δ≤1e-3, REPORT §5); queries in chunks of 64 | inferred KV-cache sharing: token counts "exactly additive" (268 → 276 tokens for 1 → 2 yes/no questions), state "counted once"; up to 1,500 questions/request tested | [AH], [TS-docs-parallel] |
| query isolation | by construction (no query↔query attention; tested) | "probability 0.00 when [secret] in sibling question, 0.90–0.92 when in state" | [AH] |
| per-decision cost model | state once + per query: 9-token suffix through 20 layers attending to L_s + O(K) MLP over cached candidate vectors | state once + per question: instruction + option text through the backbone (options are tokens, so cost grows with option text); output tokens "FREE (too cheap to meter)" | [AH], [TS-blog] |
| IIA / choice-set behaviour | independent scorer: add-irrelevant Δlog-odds = 0.000, reorder exact, duplicate mass err .047; listwise (`joint_emb_lw`): Δ = 0.168 by design | add-irrelevant option shifted log-odds "+0.38 to +0.11", mean −0.28 (95% CI −0.36 to −0.19) → IIA violated; reorder: 0.84–0.89 → 0.93–0.96; reference card last 16/16 vs ~50% first/middle | [AH] |
| training data | 758k rows / 24 tasks (SNLI, MNLI, ANLI, UNLI, BoolQ, SQuAD-v2, CLINC, HWU64, …), held-out label spaces Banking77/TREC/20NG | n/d ("Source not disclosed") | [TS-blog] |
| training recipe | soft-CE decision SFT, ~12k steps × 64, 1 epoch, AdamW; ≈1.3 H100-h ≈ $4 per run; whole project ≈ $68 (REPORT §7) | "RLCD" — named, algorithm n/d; "parallel sampler for maximum efficiency" | [TS-blog], [TS-docs-primer] |
| accuracy, shared public benchmarks | none shared. Ours: SNLI .906 / MNLI .873 / ANLI .524 / BoolQ .817 / CLINC-150 .803 / HWU64 .875 (trained); Banking77-77 .440 / TREC-coarse .210 / 20NG .273 (unseen label spaces) | only MMLU-Pro 84.6% (1,200-item sample, third party); TypeSafe workflow evals: 61.7 / 71.6 / 61.8 / 76.0% vs Opus 5 66.2 / 75.2 / 78.4 / 72.4% (Security, Agent-trace, Invoice, Customer-service) | [AH], [TS-evals] |
| calibration | ECE (val-T) SNLI .005, MNLI .016, BoolQ .049, CLINC .029, HWU64 .025; OOD ECE Banking77 .143, TREC-coarse .361, CLINC-heldout .254; ChaosNLI NLL 1.38 (worse than `mcq_lora` 1.13 / B_1.7B 1.16) | MMLU 1,200 items, 10-bin ECE 0.0313, raw API output (no fitting); 990/1,200 items in the 0.9–1.0 bin (pred 98.7%, obs 96.3%); fresh 3-digit multiplication 86.7% acc at 0.83 mean p; 2-step word problems 32% at 0.30 | [AH] |
| abstention / OOS | CLINC-OOS recall .649 (`emb`) / .823 (`emb_lw`); null AUROC clinc-k .953, snli .979, squad .976; P(∅ \| gold absent) falls .97→.31 (K 2→150; `lw`: .91→.48) | n/d (no null primitive; behaviour with a literal "none" option unmeasured) | — |
| latency (single state) | H100 SXM, bf16, 212-token state, 9-token queries, median of 5, batch = 1 state: state 22 ms; M=1 total 74 ms; M=256 total 210 / 417 / 1,139 / 6,553 ms at K = 4 / 32 / 150 / 1000 → marginal 0.73 / 1.53 / 4.36 / 25.5 ms per query | end-to-end "70ms–500ms" [TS-blog]; measured medians [AH]: 1 question 86.5 ms (short state), 100 q 82 ms, 500 q 181 ms, 1,000 q 453.5 ms, 1,500 q 610 ms → slope ≈ 0.35–0.43 ms/question (inferred from those medians; K and hardware n/d); state sweep, 1 q: 360 tok 57.5 ms → 29.8k tok 218 ms (≈ 5.4 ms per 1k state tokens, inferred slope) | [TS-blog], [AH] |
| latency vs prompted LM | ours vs prefix-sharing log-prob baseline (same 1.7B, candidates batched, queries sequential): 73× / 37× / 20× / 16× cheaper per query at K = 4 / 32 / 150 / 1000 (M=256) | "40x–200x faster" than frontier LLMs whose "end-to-end response time is … 3 to 329 seconds"; workflow evals "193.6x faster, 444.6x cheaper" (vendor caveat: "likely on the higher end") | [TS-blog], [TS-evals] |
| throughput | not measured (single stream) | rate limit "250,000 tokens per second / 1,200 requests per minute" (a quota, not a measurement) | [TS-docs-models] |
| cost per 1k decisions | derived: H100 SXM $3.00/h (PLAN2 RunPod rate), single stream, M=256 incl. state encode: **$0.0007 (K=4) / $0.0014 (K=32) / $0.0037 (K=150) / $0.021 (K=1000)**; prefix-sharing log-prob baseline $0.045 / — / $0.072 / $0.35. Not a saturated server; a batched serving stack would be lower | list price $0.042/M input tokens, output free. Derived: 212-token state + 1k questions × 8 (yes/no, measured) … ~40 (4-option choice, guessed) tokens = 8k–40k tokens → **$0.0003–$0.0017 per 1k decisions**. Datapoint: 13 questions over a ~12k-token doc = $0.000497, 0.27 s; 13 single calls = $0.00609, 2.71 s | [TS-docs-models], [TS-docs-parallel], [AH] |
| label-space generality | weak: unseen label vocabularies → false null (CLINC-heldout acc .614 vs among-K .855; Banking77-K null AUROC .745 vs .953 on trained vocab); `mcq_lora` (options in context) gets .890 / .568 / .534 on CLINC-heldout / Banking77-77 / 20NG | options are context tokens with descriptions → the mechanism `mcq_lora` has; docs' known limits: "quite literal", "accuracy falls as the state grows with content unrelated to the decision", no P(noul)+P(¬noul)=1 guarantee | [TS-docs-jagged], [AH] |
| open / verifiable | code, data (HF `guychuk/pcdm-data`), checkpoints (HF `guychuk/pcdm-runs`), W&B, `tests/` (16 CPU tests) | closed weights, closed data; API in early access; MIT adapter for LLM comparison; [AH] publishes an evidence bundle (`/research/jev/evidence.json`, 1,029+ probe records, 6,800 benchmark items) | [TS-blog], [AH] |

### 2a. Published numbers on our public benchmarks (context, not a Jev comparison)

| set | ours (`joint_emb`, K + null, runtime label strings) | published reference | note |
|---|---|---|---|
| MNLI-m | .873 | RoBERTa-large 90.2; DeBERTa-V3-large 91.8 | fixed 3-way heads, full fine-tune |
| ANLI test (R1–3 pooled) | .524 (.547 `abl_notower`) | RoBERTa-large (S+M+F+ANLI) A1 73.8 / A2 48.9 / A3 44.4 → ≈ 54.9 pooled | comparable regime (ANLI in train) |
| BoolQ | .817 | RoBERTa-large 86.9 | ≤1–2 pts inflated by near-dup leak (REVIEW §6) |
| CLINC-150 (in-scope) | .803 acc / .821 among-K | BERT-tuned 96.93; OOS recall BERT 40.3 (oos-train) / 52.3 (threshold) vs ours .649 / .823 (`lw`) | ours is 151-way with learned ∅ |
| HWU64 | .875 | BERT-tuned 92.10; USE+ConveRT 92.62 | |
| Banking77 | .440 (label space never trained) | BERT-tuned 93.66 (full); zero-shot open-weight 7B ≈ .66–.79 with the label list in the prompt | ours ≈ zero-shot regime |
| ChaosNLI-M | NLL 1.38 / JSD(nat-log) .103 | paper reports JSD/KL under different conventions | not comparable as-is |

## 3. Who is ahead, where it is not apples-to-apples, what to run

**Measurably ahead — PCDM.** (i) Choice-set invariances: reorder exact, add-irrelevant Δ = 0 (Jev: −0.28 log-odds, order flips
0.84→0.93, position-dependent accuracy 16/16 vs ~8/16). (ii) An architectural null with measured AUROC .95–.98 on three slices; Jev has
none (literal option). (iii) K-independent suffix cost: 0.73 → 4.4 ms/query from K=4 to K=150 on one H100, no 255-option cap.
(iv) Everything is open and re-runnable for ≈ $4/run.

**Measurably ahead — Jev.** (i) Scale of the state: 30k tokens in ≈ 0.2 s; we have never benchmarked past 212 tokens and IDEA2 §15
predicts O(M·L_q·L_s) growth. (ii) Per-question slope ≈ 0.35–0.43 ms at their K vs our 0.73 ms at K=4 (unknown hardware; likely
batched serving). (iii) Knowledge/reasoning: MMLU-Pro 84.6% and ECE .031 with no post-hoc fitting; we have no knowledge benchmark
and our OOD ECE is .11–.36. (iv) Label-space generality via in-context options with descriptions — the exact axis where our own
`mcq_lora` beats `joint_emb` by +27 (Banking77-77) / +28 (CLINC-heldout) / +26 (20NG). (v) Score and Noul primitives, JSON state.

**Not apples-to-apples.** No shared benchmark exists today. Their accuracy numbers are a vendor workflow suite with LLM-generated
reference labels ("average of the responses of GPT-6 Astra and Claude Fable 5.1") and one third-party MMLU-Pro sample; ours are
public NLU sets. Their latency is an API round-trip (queueing included) on unknown hardware at unknown batch; ours is a single
stream on a rented H100 with Python-level chunking. Their "40–200×" is against reasoning LLMs (3–329 s); our "16–73×" is against a
same-size prefix-sharing log-prob baseline that does not batch across queries. Their cost is a list price (margin included); ours
is raw GPU rent. Their null is a literal option; ours is learned. Model size differs ~6× (inferred).

**Evaluations that would make it sharp (≈ $70 H100 budget; API calls need early access).**

| # | run | cost | what it settles |
|---|---|---|---|
| 1 | Jev on our public eval files via `POST /v1/systemone` (Choice with `criteria` = our label strings, plus a literal "none of the above"): CLINC-150 test + OOS, Banking77-77, HWU64, TREC, 20NG, SNLI/MNLI/ANLI/BoolQ (1k each); score with `pcdm/metrics.py`. Also run `joint_emb` with the same literal-none candidate for protocol parity | ≈ $0.3 API (~7M tokens), $0 GPU | the only same-items comparison: accuracy, ECE/NLL/Brier, among-K vs null, seen vs unseen label spaces |
| 2 | `cse_*` / `ksweep_*` battery on Jev (add-irrelevant, reorder, duplicate, remove-gold, P("none") vs K 2→150) | ≈ $0.1 API | IIA Δ, reorder sensitivity, duplicate mass, K-dilution of the literal null vs our ∅ (.67 range) — head-to-head on our exact protocol |
| 3 | MMLU-Pro 1,200-item sample (from [AH] evidence bundle) on `joint_emb`: state = question, K=10 options; report acc, 10-bin ECE with their binning | ≈ $0.5 | the one number Jev has publicly; expected loss quantifies the "1.7B, no knowledge training" gap; bundle gives Jev per-item p for identical ECE/Brier |
| 4 | Replicate their latency protocol on ours: `pcdm/bench.py` with state 360 → 30k tokens (raise the 256 cap) and M 1 → 1,500, plus a *query-batched* B_fair; report ms/question slope and ms per 1k state tokens; add K=150 with option descriptions (~20 tokens) to show the K-independence claim under realistic option text | ≈ $2 | whether 0.73 ms/query and the 16–73× survive long states and a fairer baseline; the O(M·L_s) prediction |
| 5 | Backbone-scale control: `joint_emb` recipe on Qwen3-4B-Base (LoRA top 8, tap 20 → scale layer accordingly), one seed | ≈ $10–12 | whether the unseen-label/false-null gap (our known weakness) is a size effect before attributing it to the head |

## 4. Threats to our numbers under their protocol

| threat | effect |
|---|---|
| single global T fitted on our val | small for `joint_emb` (T = 0.96; raw ECE .007 vs scaled .005 on SNLI) but our ECE is reported *after* selection on val; Jev's .031 is raw API output on an unrelated benchmark. Report raw ECE alongside. OOD ECE .11–.36 is the honest number for an API user |
| seed noise | NLI ±0.7, trained large-K ±3, unseen label spaces / OOS ±5 (two seed pairs). Every held-out-space and OOS claim below 5 pts is a tie |
| 1.7B backbone, 20 layers used | Jev inferred ~10B-active; knowledge-heavy states (MMLU-Pro, invoices, security alerts) will expose this; do not read our NLI wins as "smaller beats bigger" |
| state length 212 tokens | no data past 256 tokens; per-query attention against a 30k prefix is unmeasured (IDEA2 §15); KV memory per state grows linearly |
| baseline batching | `B_fair` scores queries sequentially (candidates batched, queries not); a query-batched baseline could shrink 73× materially at small K. Their 40–200× is against reasoning LLMs, a different axis |
| candidate interface | ≤16-token label strings through a frozen 0.6B embedder; no option descriptions, no JSON criteria, no Score type. Their Choice schema is strictly richer; a benchmark built from their docs' examples would fail our loader before it fails our model |
| null protocol | ours: learned ∅ trained on synthetic gold-absent rows; theirs: literal option. Under their protocol our ∅ is unused and our literal-none behaviour is untested; under ours their literal none is an option the model was presumably trained for (base LMs were not: B_8B 96% null on OOS) |
| false-null on unseen vocabularies | CLINC-heldout .614 vs among-K .855; fix needs label-space-held-out training rows (REPORT §3d), not calibration. Any unseen-label eval vs Jev will show this first |
| eval caps / leaks | ours capped at ~3k items per set; BoolQ / HWU64 ≤ 1–2 pts inflated (near-dup audit); B_8B subsampled 300–1000 |
| their numbers | one third-party probe day; MMLU-Pro is a 1,200-item sample; latencies include queueing; workflow-eval labels are LLM-generated and vendor-run |

## 4a. Update (2026-09-22): JevBench leaderboard, our own frozen-control ladder, and latency vs generative LLMs

Sections 1–4 above are the 2026-09-18 comparison against TypeSafe/Jev from first-party docs and a third-party
black-box analysis, run on the pre-native-head `joint_emb` family. Everything below is newer, sourced from
REPORT.md §3q (JevBench harness run, 2026-09-20) and §3ab/§3ag/§3ah (scaling ladder, frozen controls, serving
latency, 2026-09-21/22), and uses the current native-head, typed-primitive architecture — not `joint_emb`.

**JevBench public-subset table (231 public ids; not a ranked leaderboard entry — see §3q's disclosures, reproduced
in every `releases/*.md` card and in `RESULTS.md`).** Majority/chance baselines: .311 standard / .284 easy / .336
hard.

| entry | standard | easy | hard | notes |
|---|---|---|---|---|
| PCDM energy `joint_emb_lw_v5` (2026-09-18 architecture) | .403 | .875 | .360 | superseded by the native head below |
| PCDM native `nc_v3` (pre-workflow-training) | .472 | .812 | .297 | the native-head starting point (§3q) |
| **Typical `typical-small` (`ts1b`, Release 1, 1.7B)** | .694 | 1.00 | .432 | released |
| **Typical `typical-medium` (`tm1b`, Release 1, 4B)** | .806 | 1.00 | .423 | released |
| **Typical `tm2` (Qwen3.5-4B, Release-1 recipe)** | .861 | 1.00 | **.495** | trained, not released |
| **Typical `tl1b` (Qwen3-14B, long-state fix + frozen-teacher KD)** | **.931** | 1.00 | .450 | candidate, not released — level with `system-one-open` on standard |
| open-jev-deberta-v3-large (classifier, local CPU) | .431 | 1.00 | .378 | closest architectural transport to ours pre-Release-1 |
| GLiNER2 / jeff (GLiFormer 400M) / Laya (ModernBERT) | .639 / .750 / .694 | 1.00 | .369 / .387 / .351 | small encoders trained on the task family |
| open-alternative-jev (Qwen3.5-4B, frozen, their rendering) | .833 | 1.00 | .568 | frozen backbone, no task training |
| system-one-open (Gemma E2B LoRA) / system-one (Qwen3-8B) | .931 / – | 1.00 | .486 / .486 | `tl1b` ties system-one-open on standard |
| SemIf (Qwen3.5-4B) / OpenJev (26B-A4B) / djev | .986 / .972 / .986 | 1.00 | .613 / .640 / .676 | still ahead of everything we've measured on hard |
| Jev 1.13.0 (closed) | .986 | 1.00 | .730 | closed reference ceiling |

The gap to the leading open entries on standard is now small (`tl1b` .931 vs SemIf/OpenJev/djev .97–.99) and the
gap on hard is where the project's own work still concentrates (`tm2` .495 is our best hard number at any size we
control; the leaders are at .61–.73).

**Our own frozen-control ladder (no training, letter logits over rendered options; §3ab/§3ag) — how much of any
frozen model's leaderboard number is backbone scale vs training vs rendering:**

| backbone | 0-shot std/hard | 3-shot std/hard | 3-shot, SemIf-style render std/hard |
|---|---|---|---|
| Qwen3-1.7B-Base | .583 / .369 | .528 / .369 | – |
| Qwen3-4B-Base | .722 / .414 | .778 / .441 | – |
| Qwen3-8B-Base | .375 / .360 (broken: all-"A" bug) | .556 / .369 | – |
| Qwen3-14B-Base | .819 / .441 | .819 / **.559** | – |
| Qwen3.5-2B-Base | – | .583 / .387 | .597 / .441 |
| Qwen3.5-4B-Base | – | .764 / .495 | .847 / .468 |
| Qwen3.5-9B-Base | – | .806 / .541 | **.931 / .595** |

Reading: instruct-tuning Qwen3-4B changes nothing on this task (§3ag); the Qwen3.5 generation is measurably better
on hard than Qwen3 at matched shot count (4B: .495 vs .441); and *rendering* — not just scale or generation —
explains a large share of the published leaderboard's standard-tier numbers: the same frozen Qwen3.5-9B checkpoint
goes from .806 to .931 standard purely by changing how the state/options are rendered, with zero training. This is
the strongest evidence that SemIf/OpenJev/djev's .97–.99 standard is partly a rendering/protocol effect, not solely
"bigger frozen model reads logits."

**Latency vs generative LLMs.** Two separate comparisons, not directly poolable (different harnesses/hardware):

- *Within the JevBench harness* (§3q): our native distributions run at p50 .13–.19 s raw (in-process, one H100,
  cold label embedding included for the energy variant) vs .17–.24 s for the GPU-hosted generative entries on the
  same leaderboard — competitive even before the serving-path optimization below.
- *Our own prompted-LM baseline, same backbone size, same hardware* (`bench_fair`, pre-native-head but architecture-
  independent for this comparison): 73× / 37× / 20× / 16× cheaper per query at K = 4 / 32 / 150 / 1000 (M = 256)
  than a same-size prefix-sharing log-prob baseline that has to re-run a forward pass per candidate — this is the
  generation-vs-direct-readout gap in isolation, not a vendor's number.
- *Public serving path, after the zero-copy KV-cache fix* (§3ag, `runs/serve_bench2/`): warm p50 **15.5–17 ms**
  (`typical-small`, 1.7B) and **19–21 ms** (`typical-medium`, 4B) per decision, down from 21–27 ms pre-fix. TypeSafe
  quote "70ms–500ms" end-to-end for Jev and "3 to 329 seconds" for the frontier LLMs they compare against
  [TS-blog] — our released models' warm decisions are below TypeSafe's own quoted floor for Jev, on different
  hardware and without a network round-trip, so treat this as directional, not a head-to-head claim.

## 5. References

TypeSafe (first party)
- [TS-blog] Introducing System One Models & Jev — https://typesafe.ai/blog/introducing-system-one-models-and-jev
- [TS-evals] Workflow evals — https://evals.typesafe.ai/
- [TS-docs-system-one] https://docs.typesafe.ai/concepts/system-one
- [TS-docs-choice] https://docs.typesafe.ai/primitives/choice.md
- [TS-docs-models] Models, pricing, limits, rate limits — https://docs.typesafe.ai/models.md
- [TS-docs-confidence] https://docs.typesafe.ai/confidence.md
- [TS-docs-primer] https://docs.typesafe.ai/introduction/machine-learning-primer.md
- [TS-docs-parallel] Batching questions cookbook — https://docs.typesafe.ai/cookbooks/parallel_questions.md
- [TS-docs-jagged] Known limitations of Jev 1.13 — https://docs.typesafe.ai/model-jaggedness/jev-1.13.md
- Doc index — https://docs.typesafe.ai/llms.txt
- System One adapter (MIT; LLM emulation of the interface, source of the confidence formula) — https://github.com/typesafe-ai/system-one-adapter-python

Third party on Jev
- [AH] Archer Hume, "Jev's architecture, unmasked" (17 Sep 2026) — https://archerhume.com/posts/jevs-architecture-unmasked ; evidence bundle https://archerhume.com/research/jev/evidence.json

Prior art for "encode once, many decisions, runtime candidates"
- Poly-encoders (Humeau et al., ICLR 2020) — cached context codes + late attention over candidates — https://arxiv.org/abs/1905.01969
- DeFormer (Cao et al., ACL 2020) — decompose lower layers, precompute the passage — https://arxiv.org/abs/2005.00697
- PreTTR (MacAvaney et al., SIGIR 2020) — precomputed document term representations for cross-encoders — https://arxiv.org/abs/2004.14255
- ColBERT (Khattab & Zaharia, SIGIR 2020) — late interaction — https://arxiv.org/abs/2004.12832
- LUMEN (de Jong et al., ICML 2023) — pre-computed memory vs on-the-fly encoding hybrid — https://arxiv.org/abs/2301.10448
- Prompt Cache (Gim et al., 2023) — modular KV reuse across prompts — https://arxiv.org/abs/2311.04934
- SGLang / RadixAttention (Zheng et al., 2023) — automatic shared-prefix KV reuse in serving — https://arxiv.org/abs/2312.07104
- Batch Prompting (Cheng et al., 2023) — many questions per LLM call — https://arxiv.org/abs/2301.08721
- Zero-shot text classification as entailment with runtime label names (Yin et al., EMNLP 2019) — https://arxiv.org/abs/1909.00161
- Calibrate Before Use (Zhao et al., ICML 2021) — LLM label-token probabilities and their miscalibration — https://arxiv.org/abs/2102.09690
- Label-Supervised LLaMA (Li et al., 2023) — decoder LLM as classifier with a head — https://arxiv.org/abs/2310.01208
- GLiNER (Zaratiana et al., 2023) — labels and text jointly encoded, runtime label sets — https://arxiv.org/abs/2311.08526
- Selective classification / abstention (Geifman & El-Yaniv, NeurIPS 2017) — https://arxiv.org/abs/1705.08500
- Kimi Linear / KDA (Moonshot, 2025) — fixed-size recurrent state for the IDEA2 §14–19 scaling path — https://arxiv.org/abs/2510.26692

Benchmarks and published numbers
- Casanueva et al. 2020 (Banking77 / HWU64 / CLINC150; BERT-tuned 93.66 / 92.10 / 96.93; USE+ConveRT 93.36 / 92.62 / 97.16) — https://arxiv.org/abs/2003.04807
- Larson et al. 2019 (CLINC150 + OOS; BERT in-scope 96.9, OOS recall 40.3 oos-train / 52.3 oos-threshold) — https://arxiv.org/abs/1909.02027
- Nie et al. 2020 ANLI (RoBERTa-large A1 73.8 / A2 48.9 / A3 44.4) — https://arxiv.org/abs/1910.14599
- Nie et al. 2020 ChaosNLI — https://arxiv.org/abs/2010.03532
- Chen et al. 2020 UNLI — https://arxiv.org/abs/2004.03744
- Clark et al. 2019 BoolQ — https://arxiv.org/abs/1905.10044 ; RoBERTa (Liu et al. 2019; MNLI 90.2, BoolQ 86.9) — https://arxiv.org/abs/1907.11692
- DeBERTaV3 (He et al. 2021; MNLI-m 91.8) — https://arxiv.org/abs/2111.09543
- Zero-shot open-weight LLMs on intent classification (2026; label list in prompt, free-text exact match; fine-tuned encoders lead zero-shot by ≈10–25 pts) — https://arxiv.org/abs/2607.27421
- SNLI leaderboard — https://nlp.stanford.edu/projects/snli/
