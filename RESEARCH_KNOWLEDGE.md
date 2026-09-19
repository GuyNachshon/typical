# Parametric knowledge for PCDM without generation — research memo (2026-09-19)

Question: how to get MMLU-Pro-class knowledge/reasoning into PCDM while keeping its cost profile (state encoded once,
queries as cheap KV-cached suffixes, calibrated softmax with a learned null, no token generation). Grounded in REPORT.md §1/§3c–§3f,
COMPARE.md, IDEA2.md, `encode.py` / `model.py` / `mcq.py` / `baselines.py` / `scripts/mmlu_pro_eval.py`. Web claims are cited inline; every number
we did not measure ourselves carries a URL in §5.

## 1. Diagnosis — why we are at chance (8–9 %, among-K 13 %, ECE .3)

Four stacked causes, in order of size. **(a) The options never enter the backbone.** `scripts/mmlu_pro_eval.py` puts the question in the
state and the ten option strings into the *frozen Qwen3-Embedding-0.6B* path; the scorer sees `h ⊙ emb(option)`. Every mechanism the literature
finds for MCQ answering — symbol binding in 1–3 middle layers [Wiegreffe 2025], "select-and-copy" heads at 40–60 % depth that compare the last
token's query with each option's end-token key [Tulchinskii 2024], and the linear knowledge probes of [Bridging the Knowledge-Prediction Gap 2025] —
reads *option tokens that were processed in context with the question*. An embedding of "3.2 × 10⁵ J" carries no information about whether it
follows from the stem; the head can only learn evidence↔label fit, which is exactly what our 24 training tasks are. Jev does read options in context:
adding an irrelevant option shifts the odds of the others, a reference card placed *after* the options is read 16/16, and cost grows with input tokens
(archerhume). **(b) No knowledge-MCQ training signal at all** (REPORT §3f reason 1) — every training row is "answer is in the state". **(c) Truncation at
layer 20/28 is a risk, not a proven killer**: MCQ symbol binding localises at ~75 % depth in OLMo (layer 24/32), knowledge probes "saturate at higher
layers", but select-and-copy heads peak at 40–60 % depth; the answer is a per-layer probe curve ($2, §3 E1), not a redesign. **(d) The novelty ⇒ null
confound** (§3d) fires on the unfamiliar option vocabulary (among-K > acc). Expectation-setting: a *no-generation* readout tops out at the model's
direct-answer accuracy, not its CoT accuracy — MMLU-Pro is built to need CoT (GPT-4o 53.5 direct vs 72.6 CoT; Llama-3-8B 31.5 vs 35.4; Gemma-7B 27.0 vs
33.7 [MMLU-Pro paper]), and linear probes recover ≈ generation accuracy on MMLU-STEM (+0.2 pts for Qwen2.5-7B) rather than hidden extra knowledge.
Qwen3 base MMLU-Pro (5-shot **CoT**): 0.6B 24.7 / 1.7B 36.8 / 4B 50.6 / 8B 56.7 / 30B-A3B 61.5 [Qwen3 report]. Jev's 84.6 is above GPT-4o's
direct-answer 53.5 while scoring 32 % on fresh two-step word problems (archerhume) — read it as in-distribution training on MMLU-Pro-style MCQ, not as
latent reasoning; NVIDIA's OpenScience is a public 6M-item, decontaminated, 10-choice synthetic set built for exactly that purpose.

## 2. Options

Per-query cost today: 0.73 ms marginal at K=4 (9-token suffix, 20 layers, M=256 on one H100; `bench_fair`). Cost multipliers below are
token-count × layer-count estimates against that. "MMLU-Pro" = expected no-generation accuracy on our 1,200-item sample; comparables in brackets.

| approach | what changes in our stack | expected MMLU-Pro | per-query cost vs today | training $ (H100 $3/h) | risk |
|---|---|---|---|---|---|
| A0. as built (embedder candidates, layer 20, no MCQ data) | – | **8–9 % measured** | 1× | 0 | – |
| A1. B-path cloze log-probs (K option continuations vs shared prefix), 1.7B/8B full depth | eval-only, `baselines.py B` | 1.7B ≈ 18–25 %; 8B ≈ 30–40 % [cloze ≤ options-in-context DA; Llama-3-8B DA 31.5] | 53 ms sequential today (`B_fair`), batched ≈ K·L_opt/9 ≈ 10–15× | $0.3 | no listwise view, T≈4 miscalibration (ECE .4–.5), K continuations |
| **A2. Options-in-suffix "pointer tier"**: suffix = query + option lines + decide token; `C_j` = backbone state at option j's end token (query- and state-conditioned) fed to the *existing* scorer + candidate-aware null; label tasks keep the cached-embedding tier | `mcq.py`-style suffix builder + gather of option end positions → `C`; train on MMLU-aux + OpenScience-10 + existing mix | 1.7B full depth: **28–34 %** [1.7B CoT 36.8; DA gap −4…−8 pts; head-tuning beats label generation +2–3 at 0.6B/1.7B (Punching Above Their Weight)]; kev (0.5B, same readout) 80 % in-dist. on NLU tasks | ≈ (L_q + Σ L_opt)/9 ≈ **10–12×** (≈ 8 ms at 28 layers; MMLU-Pro ≈ 100 option tokens); still one pass, no vocab projection, no K continuations; large-K label tasks stay 1× on the cheap tier | $5–8 (options lengthen sequences; MCQ steps were ~4× slower in `mcq_lora`) | order effects (Jev shows them; randomise order in training, report reorder Δ); null must be re-taught for the new tier |
| A3. A2 + full depth (tap 28 instead of 20) | `--tap_layer 0`, LoRA on 21–28 | +0…+5 over A2 (unknown; E1 decides) | +40 % layers (28/20) | same | 8 more layers of KV per state (112 vs 80 KB/token) |
| A4. A2 on Qwen3-4B-Base (36 layers, d 2560) | `--backbone`, LoRA top 8–10 | **40–46 %** [4B CoT 50.6; DA ≈ −5…−8] | ≈ 3× A2 (FLOPs/token) → ≈ 25 ms/query; KV 147 KB/token (30 k-token state = 4.4 GB) | $10–14 | per-query cost now within 2× of a batched B-path; 8 GB weights |
| A5. A2 on Qwen3-8B-Base | same | 48–52 % [8B CoT 56.7] | ≈ 6.5× A2; KV 147 KB/token | $20–25 | over budget, 16 GB weights, slope worse than Jev's ~0.4 ms/question |
| A6. A2 on Qwen3-30B-A3B-Base (MoE, 3B active) | same; needs 61 GB bf16 weights on the GPU | 52–56 % [30B-A3B CoT 61.5] | compute ≈ 2.5× A2 but expert dispatch for a 100-token suffix is bandwidth-bound; KV **98 KB/token** (4 KV heads) — best KV/token of the set; suffix cost stays K-independent (routing is per token) | $30+ (training MoE LoRA at 61 GB + activations on one 80 GB H100 is marginal) | memory, engineering; not for this budget |
| A7. Looped backbone (Ouro-1.4B: 24 layers × 4 recurrent steps, open weights) | new modeling code (`OuroForCausalLM`), LoRA hooks inside the loop, 4 KV caches per state (MHA, 16 KV heads: 196 KB/token/step → 786 KB/token; "last-step reuse" 196 KB) | 1.4B: CoT 48.6 vs Qwen3-1.7B 37.3, 2.6B 55.7 vs Qwen3-8B 53.7 [Ouro]; DA unknown | ≈ 96 layer-passes vs 20 → ~5× A0 per token, ×11 tokens ≈ 50× | $8–10 once code works | custom code path, 49k-vocab tokenizer, KV memory 7–10× ours; the "knowledge manipulation" gain was measured with CoT |
| A8. Add latent steps to the suffix (pause tokens, Coconut, KV-cache coprocessor, retrofitted recurrence) | new training objective | Gemma-2-2B coprocessor: MMLU +4.7 with 64 latents after 100 k steps × 1,024 × 2,048 tokens ≈ 2×10¹¹ tokens; retrofitted recurrence: ~100 B tokens, MMLU unchanged, GSM8K +5; pause tokens help only if in pretraining; Coconut ≤ CoT on GSM8K (34.1 vs 42.9); training-free looping of Qwen3-4B-Instruct: MMLU-Pro +2.6 | +latents/loops per query | ≥ $10 k for the trained variants; $0 for training-free (+2.6) | not fundable; the gains reported are on math, not MMLU |
| A9. Distil System 2 → System 1 (teacher CoT, student answers directly) | data only (labels), same head as A2 | works where the task is in-distribution (Yu 2024: 30 → 98 on last-letter, S2A 51.6 → 81.3), retained ~75 % of CoT on GSM8K for implicit-CoT Mistral-7B (0.52 vs 0.68 explicit); for MMLU-Pro this *is* A2's training set | 0 | teacher labels: OpenScience is free (teacher = DeepSeek-R1 / Qwen3-235B traces); API soft labels at GPT-5-nano batch $0.025/$0.20 per M: 100 k items × (400 in + 300 out) ≈ **$7**; GPT-5-mini batch ≈ $35; with 1.5 k-token CoT ≈ $30 / $150 | teacher calibration transfers to the student (Role of Teacher Calibration in KD 2025) — use logprob-bearing teachers or self-consistency counts, not verbalised confidences |
| A10. Retrieval as state (Wikipedia passage in the state) | data pipeline; head unchanged | lifts factual-recall MMLU for ≤ 3B models, gain shrinks by 32B and is small on MMLU-Pro/ARC-C (RAG in the Wild 2025) | state longer (+500–2 k tokens), suffix unchanged | $0 model, retrieval infra | orthogonal; only helps the recall slice, not the numeric/reasoning slice |

Reading: **A2 is the only change that fixes cause (a)**, and it is what Jev's black-box signature and the open re-implementation (kev: `</opt>`
states scored against a `<decide>` state, block-causal branches, CE loss, T-scaled ECE .031) both point to. It is *not* a two-tier "LM-scored
tier" — Jev's 255-option cap, additive token accounting and per-question slope (0.3–0.4 ms) fit one sequence per question with a pointer/slot readout,
not K continuations. Our cheap cached-embedding tier is something Jev lacks (K-independent, 150-way intents at 0.7 ms); we keep it and add the pointer
tier for option sets that must be *read* (numbers, clauses, unseen vocabularies — where `mcq_lora` already beat us by +27 on Banking77-77).

## 3. Experiment ladder (≈ $25 of the $40)

| # | run | $ | what it isolates | pre-registered success rule |
|---|---|---|---|---|
| **E0** (already queued) | `B_8B_mmlu`, plus `B_1.7B_mmlu` on the same 1,200 items | $0.5 | backbone knowledge under cloze scoring vs our head | record only. Prediction: 1.7B 18–25 %, 8B 30–40 %; if 8B ≥ 50 % the cloze prior is worth a feature, else drop A1 |
| **E1** frozen per-layer pointer probe | encode 30 k OpenScience-10 + MMLU-aux rows *with options in the sequence* once (`output_hidden_states`), keep layers {12,16,20,24,28} at each option's end token and at the decide token; fit the existing scorer (no LoRA) per layer; eval on MMLU-Pro 1,200 | $2 | (a) vs (c): does reading options in context lift us off chance, and is layer 20 the wrong tap | success if any layer ≥ 20 % (2× chance) — proves the readout; choose tap = argmax layer; if layer 28 − layer 20 ≥ 4 pts, go full depth in E2. Failure (all layers < 15 %) ⇒ knowledge readout needs LoRA, still run E2 but with tap 28 |
| **E2** pointer tier, 1.7B, LoRA, tap from E1 | A2 training: existing mix + 100 k MMLU-aux + 200 k OpenScience-10 (hard labels, option order randomised, 15 % gold-dropped rows with p_null=1), 6 k steps; eval full suite + MMLU-Pro | $6–8 | (b): training signal; keeps all existing metrics on the cheap tier | MMLU-Pro ≥ 28 % *and* ECE ≤ .08 after T *and* no existing metric below its seed floor (NLI −0.7, large-K −3, unseen −5); reorder Δacc ≤ 2 pts. 28 % = Llama-3-8B-DA-class from a 1.7B, i.e. the readout retains ≥ 75 % of 1.7B's CoT number |
| **E3** same recipe on Qwen3-4B-Base | `--backbone Qwen/Qwen3-4B-Base`, tap ≈ 26–28 of 36 or full, LoRA top 8 | $10–14 | scale of the knowledge tier under our cost model | MMLU-Pro ≥ 40 % and per-query ≤ 30 ms at K=10 with 100 option tokens (bench); else the 4B is not worth 3× per query and we stay at 1.7B + better data |

Order is fixed: E1 before E2 (tap choice and go/no-go for $2), E2 before E3 (recipe must work at 1.7B first; `main_4B` was cut once already for
this reason). If E2 passes but ECE > .08, spend the $7 on GPT-5-nano top-logprob soft labels for the 100 k MMLU-aux rows before E3.

## 4. Do not

- Do not train latent-reasoning add-ons (pause tokens, Coconut, KV-cache coprocessor, retrofitted recurrence): every reported gain needed
  10¹¹ tokens of continued pretraining and was measured on math, not MMLU.
- Do not buy API CoT labels before OpenScience/MMLU-aux (free, decontaminated, 10-choice) are exhausted; never train on MMLU test or MMLU-Pro test.
- Do not try to fix MMLU-Pro through the cached-embedding tier (bigger embedder, listwise mixer, null bias): the option text must reach the backbone.
- Do not read Jev's 84.6 as the bar for a no-generation model of any size; GPT-4o direct-answer is 53.5 and Jev is 32 % on fresh two-step problems.
- Do not run 8B dense or 30B-A3B on this budget; do not switch to Ouro/KDA backbones until the recipe works on stock Qwen (IDEA2 §18 rule).
- Do not ship the B-path cloze scorer as the knowledge tier: K continuations, no listwise view, ECE .4–.5 — E0 is a measurement, not a design.

## 5. References

- Qwen3 technical report (base MMLU-Pro, 5-shot CoT) — https://arxiv.org/html/2505.09388v1
- MMLU-Pro (CoT vs direct answer; 10 options; 12,032 items) — https://arxiv.org/html/2406.01574v4
- Bridging the Knowledge-Prediction Gap in LLMs on MCQ (linear knowledge probes; MMLU-STEM +0.2, TruthfulQA +21) — https://arxiv.org/html/2509.23782v3
- Answer, Assemble, Ace (symbol binding in middle layers, ICLR 2025) — https://arxiv.org/abs/2407.15018
- Listening to the Wise Few: select-and-copy heads for MCQA (layers 12–21 of 32; +16 on LLaMA2-7B) — https://arxiv.org/html/2410.02343
- LLMs Know More Than They Show (truthfulness at exact-answer tokens; ICLR 2025) — https://arxiv.org/abs/2410.02707
- The Unreasonable Ineffectiveness of the Deeper Layers (MMLU flat until 40–50 % pruned; final layer special) — https://arxiv.org/abs/2403.17887
- Punching Above Their Weight: classification-head fine-tuning of Qwen3-0.6B/1.7B for MCQ — https://arxiv.org/pdf/2607.03801
- Robinson & Wingate, multiple-choice vs cloze prompting — https://arxiv.org/pdf/2210.12353
- Distilling System 2 into System 1 (Yu et al. 2024) — https://arxiv.org/abs/2407.06023v2
- From Explicit CoT to Implicit CoT (Deng et al. 2024; Mistral-7B 0.52 vs 0.68) — https://arxiv.org/pdf/2405.14838
- Coconut (Hao et al. 2024; GSM8K 34.1 vs CoT 42.9) — https://arxiv.org/html/2412.06769v1
- Think Before You Speak: pause tokens (gains need pretraining with pauses) — https://arxiv.org/abs/2310.02226
- Deliberation in Latent Space via Differentiable Cache Augmentation (Gemma-2 2B; MMLU +4.7 @ 64 latents) — https://arxiv.org/html/2412.17747
- Scaling up Test-Time Compute with Latent Reasoning (Huginn-3.5B; MMLU 23.4 → 31.4 over 4 → 32 recurrences) — https://arxiv.org/pdf/2502.05171
- Retrofitted Recurrence (≈100 B tokens; MMLU unchanged) — https://arxiv.org/html/2511.07384v1
- Training-Free Looped Transformers (Qwen3-4B-Instruct MMLU-Pro +2.6) — https://arxiv.org/abs/2605.23872
- Ouro: Scaling Latent Reasoning via Looped LMs (1.4B MMLU-Pro 48.6, 2.6B 55.7; HF ByteDance/Ouro-1.4B) — https://arxiv.org/html/2510.25741
- The Role of Teacher Calibration in Knowledge Distillation — https://arxiv.org/html/2508.20224
- Trust the Uncertain Teacher (calibrated-uncertainty distillation) — https://arxiv.org/html/2602.12687v2
- NVIDIA OpenScience (6M synthetic MCQ, 4/10-choice splits, decontaminated vs MMLU/MMLU-Pro/GPQA) — https://huggingface.co/datasets/nvidia/OpenScience
- RAG in the Wild (retrieval lift vanishes with backbone size on MMLU/MMLU-Pro) — https://pith.science/paper/2507.20059
- OpenAI API pricing 2026 (GPT-5-mini $0.25/$2.00, nano $0.05/$0.40 per M; batch ½) — https://www.getapipulse.com/blog-gpt5-mini-cost-breakdown.html
- Archer Hume, Jev's Architecture Unmasked — https://archerhume.com/posts/jevs-architecture-unmasked/
- jaredpalmer/kev (open Jev-style pointer head on Qwen2.5-0.5B) — https://github.com/jaredpalmer/kev
- HF configs used for KV/token: Qwen3-1.7B/4B/8B/30B-A3B-Base, ByteDance/Ouro-1.4B/2.6B — https://huggingface.co/Qwen/Qwen3-30B-A3B-Base , https://huggingface.co/ByteDance/Ouro-1.4B
