# Stop-and-rethink review (2026-09-18) — after `joint_v1`

Inputs: IDEA2.md (PI's revised statement), REPORT.md, all `runs/*/results.json`; four reviewers — fable (scientific audit),
opus (Phase-2 engineering spec), sonnet (literature verification), Codex gpt-5.5 (adversarial). Full memos are in the
session transcript; this file is the synthesis and the decisions.

## 1. What stands

- **Joint prefix encoding is the architecture.** SNLI 91.0 / MNLI 87.4 / ANLI 54.4 / BoolQ 81.5 vs 70.1 / 64.8 / 43.5 / 66.3
  for the separate-encoder tower; the 4×1024-d tower control is *worse* (64.1). Hypothesis-only probe 44%. Real-model
  KV-cache path ≡ concatenation path (Δ ≤ 1e-3; chunk sizes 1–64; permutation-consistent; repeat-exact). Eval path is
  clean (best-by-val, one T fitted on val). Literature (DeFormer, PreTTR, LUMEN, Poly-encoders) says exactly this.
- Query isolation and direct probabilities hold by construction.
- In-distribution calibration: ECE .016–.037 on SNLI/MNLI/BoolQ/CLINC/HWU64.

## 2. What does not stand yet (and why)

| claim | verdict | evidence |
|---|---|---|
| "beats prompted LMs on large-K / null" | **unproven — baseline broken** | `baselines.py` B scores each label as a continuation of `Text: …\nLabel:` with no query and *no other options*; literal "none of the above" over/under-fires (B_8B: 2% on HWU64, 96% null on OOS). Subsampled 300/1000; T fitted on our val. |
| "explicit unknown, not a global OOD score" (IDEA2 §2.5) | **contradicted** | Training couples K and null: K≥50 rows never null, NLI K=2 rows 50% null, K=N never null. Result: `clinc_oos` at K=150 flagged 32%; null over-fires on unfamiliar label sets (TREC acc 5% vs 27% among-K; paraphrase 62 vs 81). The .90/.98/.97 AUROCs are measured where the training prior matches. |
| "general decisions" on unseen label spaces | **loses to the prompted LM among-K** | banking77_k 0.61 vs 0.69 (B_8B); 20NG 0.22 vs 0.50; TREC-coarse 0.27 vs 0.45. Frozen cosine is the entire transfer mechanism (`abl_nohybrid`: Banking77 33 → 8); candidate encoder (mean-pooled layer-12) is the likely bottleneck. |
| "calibrated uncertainty" | **in-distribution only** | OOD ECE .19–.69 (Banking77 .29, TREC .69); premise-blanked NLL 1.37 > uniform 1.10 — confidently wrong without evidence. ChaosNLI: joint 1.26 vs B_1.7B 1.16 (sharper on disputed items — expected per Nie et al. 2020; report by agreement subset). |
| H2 for the joint model | **unmeasured**; prior 19–69× was tower vs a B that did not share the prefix | expect single-digit × at K=4; must bench with a fair B (state prefix shared). |
| seeds | none clean | ±1 pt was data-v2 vs data-v3, not a seed pair. `joint_v1_s1` running. |

Literature notes (sonnet): Kimi Linear numbers verified; the Qwen→KDA "interface injury" result is real (arXiv 2608.02689: MCQA
collapses to ~25% at teacher-level perplexity) — KDA stays deferred; the five-slice null taxonomy and K-dependence of abstention
are *original* claims, not prior art; focal loss / Dirichlet calibration are cheap Stage-B alternatives; IDEA2's RLCD framing
(TypeSafe: named but undisclosed) is accurate.

## 3. The "fancy classifier over Qwen" question — resolved into an experiment

After joint encoding, `joint_v1` = causal cross-encoder over (state, query) with a label-embedding head and a learned reject.
Whether that head matters is answered by **MCQ-LoRA**: identical backbone/LoRA/data/steps, options enumerated in the suffix
(`A. …`, …, `Z. none of the above`), readout = next-token letter logits, soft-CE with the "none" letter carrying p_∅.
If MCQ-LoRA ≥ joint_v1 − noise on SNLI/MNLI/CLINC-150/HWU64/Banking77-among-K/null AUROC/ECE, the energy head is decorative at
K ≤ 150 and PCDM's residual claim is K-independent cost — which must then be measured, not assumed. Prediction: MCQ-LoRA ties
on NLI, *wins* on unseen label spaces (labels attend to state and each other — listwise for free), loses on cost at large K.
`abl_notower` (running) only asks whether 2 cross-attn layers beat mean-pooling; it cannot answer this.

## 4. Decisions

Kept: `abl_notower` (running), `joint_v1_s1` (running), cost guard. **Cut:** `joint_24k` (answers no hypothesis), 4B, KDA (H6),
Stage C (H7) except a $0 utility evaluation. IDEA2 wording to change: "reusable neural memory" → "prefix KV cache";
"decision compression" → "retention under fine-tuning"; H2 "strongly supported" → "unmeasured for the joint model";
§2.5/§2.6 → state the measured in/out-of-distribution behaviour.

## 5. Plan (≈ $45 of H100; ranked by information per dollar)

| # | item | $ | falsifies |
|---|---|---|---|
| 1 | cosine-only label matching on held-out spaces with 3 candidate encoders (layer-12 pool, layer-20 pool, sentence embedder); no training | 0 | "the learned scorer helps transfer" if untrained cosine ≥ 0.31 among-K on Banking77; "listwise is the fix" if an embedder beats layer-12 by >10 |
| 2 | `bench_joint` with a **fair B** (state prefix shared, candidates batched) | 1 | H2-joint if marginal advantage < 3× at K=4 / < 8× at K=150 |
| 3 | `joint_v1_s1` (running) | 4 | establishes the noise floor |
| 4 | **MCQ-LoRA** (§3) | 8 | the decision-head thesis |
| 5 | zero-shot 8B with enumerated options + "none" letter, uncapped | 3 | "learned null > literal null" |
| 6 | **data v4**: decouple K from p_∅ (gold-absent at every K incl. K=N; flat null rate across K); sibling near-miss nulls; duplicate / cross-family distractor augmentation; evidence-removal rows (Stage-B-lite); new eval sets: `cse_*` battery (add-irrelevant / remove-gold / duplicate / reorder / near-dup), `ksweep_*` (P(∅|absent) vs K), `null_irrq_*`, `null_nearmiss_*`; null metrics reported per K | 5 build + 4 train (`joint_v2`) | "null is architectural, not a K prior" if `clinc_oos`@K=150 stays < 0.6 after decoupling |
| 7 | `joint_lw` = `joint_v2` + `--listwise` (opus spec: zero-init set mixer, +2.1M, exact-equivariance tests) | 4 (+4 seed) | H5 per the pre-registered rules (p_null_absent range over K ≤ 0.10; dup mass error ≤ 0.05; +3 Banking77-k, +2 CLINC-heldout; no NLI regression) |
| 8 | `--dump_probs` + expected-cost evaluation under an asymmetric cost matrix (H7 precursor) | 0.5 | whether calibration differences change decisions at all |

Order: 1 → 2 → 4 → 5 → 6 → 7 → 8 (3 in flight). Stop rule: if #4 ties or wins everywhere at K ≤ 150, do not build #7 on the
energy head — re-plan around the letter readout with the cost measurement from #2.

### 5a. Result of plan item 1 (cosine probe, `scripts/cosine_probe.py`, $0)

| among-K, untrained cosine | layer-12 pool (current candidates) | layer-20 pool | Qwen3-Embedding-0.6B (`intent: {label}`) | joint_v1 trained | B_8B |
|---|---|---|---|---|---|
| banking77_k | 0.39 | 0.12 | **0.79** | 0.61 | 0.69 |
| banking77_test (K=77) | 0.16 | 0.00 | **0.53** | 0.31 | 0.45 |
| clinc_heldout | 0.42 | 0.14 | 0.85 | 0.74 | **0.88** |
| ng20_test | 0.13 | 0.08 | 0.33 | 0.22 | **0.50** |
| trec_coarse | 0.34 | 0.28 | 0.39 | 0.27 | **0.41** |

Two conclusions: (a) the trained scorer *does* add value over the raw frozen cosine (0.39 → 0.61), so "the learned scorer helps
transfer" is not falsified; (b) the candidate encoder is the bottleneck — mean-pooled decoder states are poor label embeddings; an
off-the-shelf embedder beats our trained model on every held-out label space. Decision: add `--cand_encoder qwen3emb` (candidates,
and the state/query vectors for the similarity features, from the embedding model; all cached per string) and run it as `joint_emb`
alongside `joint_v2`/`joint_lw`. Listwise stays in the plan but is no longer the first bet for H4.

### 5b. Outcomes of the plan (2026-09-18, all runs complete; details in REPORT.md §3c)

| # | item | outcome |
|---|---|---|
| 1 | cosine probe | candidate encoder is the bottleneck → `joint_emb` |
| 2 | fair bench | H2 holds fairly: 73× / 37× / 20× / 16× at K = 4 / 32 / 150 / 1000 |
| 3 | seed | ±0.7 pt NLI, ±1 large-K, ±3–5 held-out label spaces |
| 4 | MCQ-LoRA | head wins in-distribution + null + cost; MCQ wins unseen label spaces (+27–28) and paraphrases |
| 5 | zero-shot 8B letters | at chance (base model); log-prob B_8B stays the prompted reference |
| 6 | data v4 + `joint_v2` | null cleaner (OOS 72%, ECE .007, new slices .91–.99) but K-dilution is structural (range .63) |
| 7 | `joint_lw` | K-dilution .63 → .37, OOS 85%; −3–7 on large-K; IIA broken by design. H5 partial |
| 7b | `joint_emb_lw` | on embedder candidates: OOS +17/+22, range .67 → .43, CLINC-150 −4/−7 (all > 2× seed noise); unseen-label gains within ±5 seed spread. Default for null-critical use |
| 9 | `joint_emb_lw_v5` (data v5: 30 vocabularies × ≤4k rows) | unseen-label gap is mostly data: Banking77-77 +9, TREC-fine +13, CLINC-heldout +7, ANLI +3.4; null side −9 OOS. REPORT §3e |
| 3b | `joint_emb_s1` | seed floor for the emb family: NLI ±0.7, trained large-K ±3, unseen spaces / OOS ±5 |
| — | `joint_emb` | best model: CLINC 80 / HWU64 88 / paraphrase 81 / Banking77-77 44 / ECE .005; residual gap = false null |
| 8 | utility eval | not run (needs per-row probs; `--dump_probs` not implemented) |
| — | `abl_notower` | tower removed: +10 CLINC, +33 OOS recall, equal NLI |

Stop-rule check (§5): MCQ-LoRA did **not** tie/win everywhere at K ≤ 150 — it lost on every in-distribution set and on null —
so the energy head stays; the plan's next lever is closing the false-null gap on unseen label spaces (K-aware null bias fitted
on val; emb + listwise), not replacing the readout.

**K-aware null bias — falsified as a fix (REPORT.md §3d).** Val-fitted (α, β, T) = (0, −0.25, 0.96): a no-op. Fitting on
unseen label spaces gives β = +2 (Banking77-K) or β = −6 (TREC): the sign depends on whether the fitting set has gold-absent
rows. The gap is unfamiliar-vocabulary ⇒ null, not K ⇒ null; it needs label-space-level held-out training rows, not calibration.

## 6. Threats to validity still open
Baseline fairness (#2, #4, #5 close it); seed count (#3, two seeds for any claim < 3 pts); K–null coupling in data (#6);
label-name overlap between training intent sets and "held-out" Banking77 (audit by exact label-string intersection);
near-duplicate premises — **audited** (`scripts/leak_audit.py`, MinHash char-5-gram @0.8): SNLI/MNLI/ANLI pair-level near-dups 1/3/0 rows → the NLI headline is not memorisation; BoolQ 4.9% and HWU64 5.4% near-dup states leak (Wikipedia-revision twins / reused utterances) → pruned in data v4, treat current BoolQ/HWU64 as ≤ 1–2 pts inflated; single global T flatters in-distribution ECE.
