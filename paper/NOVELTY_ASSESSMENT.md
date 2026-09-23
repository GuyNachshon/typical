# Novelty Assessment

Blunt per-finding audit. Verdicts: **contribution** (defensible as new), **contribution (narrow)**
(new but scope-limited), **confirmation** (known result, new setting), **not a contribution** (must
not be claimed as a finding in the paper). Evidence pointers are to `REPORT.md` sections. Citation
keys resolve in `paper/references.bib`.

---

## F1. Candidate-blind decision states transfer priors and calibration, not question-conditioned knowledge — at any depth

**Evidence.** §3j (six factorisations/objectives: single vector `z1`, R=8 probes `zr`, set-conditioned
`zr_set`, ordinary KD, same-Z multi-set KD, explicit Δ-log-odds supervision; choices-only and
shuffled-question controls), §3s (same negative at full depth, Δ_q_sh = −.008 vs teacher .102).

**Closest prior work.** [wong2026models] — the decision is formed *after* the options have been read,
which predicts our negative mechanistically. [balepur2024artifacts] — the choices-only control we
adopt. [hinton2015distillation, kim2016seqkd, snell2022contextdistillation] — what distillation
transfers. [yuan2020teacherfree, zhang2020selfdistillation] — KD as label smoothing, i.e. why
calibration transfers cheaply. [khattab2020colbert, cao2020deformer, dejong2023lumen] — the
candidate-independent-precomputation family this refutes for knowledge-heavy decisions.
[zhao2021calibrate] — candidate priors as a first-order effect.

**Genuinely new.** The decomposition itself, stated as a measurable: *candidate priors and calibration
distil into a candidate-blind state; question-conditioned parametric knowledge does not*, established
with a control (Δ_q against a shuffled question) that separates the two, held across six head/objective
variants and both taps. We could not find prior work that runs this factorisation with this control.

**Reproduction / confirmation.** That option-set artifacts dominate weak MCQ models
[balepur2024artifacts]; that late-interaction loses on knowledge-heavy retrieval; that KD transfers
calibration.

**How a reviewer attacks it.** (i) One backbone family, 1.7B, 12k steps — "you under-trained a small
model" is the cheapest rebuttal and we cannot refute it with an impossibility argument. (ii) The
teacher's own question-dependent signal is small (Δ_q ≈ .088–.102 on MMLU-Pro), so the measurement
has ~9 points of dynamic range on n = 1,200 items (SE ≈ 1 pt); a sceptic will call the negative a
low-power measurement. (iii) The set-conditioning path's cross-attention stayed inert at γ = 1 —
an optimisation failure is an alternative explanation to an information-theoretic one, and we say so.
(iv) No result with a large or instruction-tuned teacher. **Verdict: contribution (narrow).** Must be
stated as "under these factorisations, objectives and budget", never as an impossibility.

## F2. A direct contextual readout preserves the letter interface's knowledge with a quarter of its order fragility

**Evidence.** §3l (N1 vs N2 vs N3), §3v (matched seed pair), §3r/§3t (at tap 20).

**Closest prior work.** [wong2026models] and [lieberum2023circuit] — the latent winner and the binding
step. [zheng2024selectionbias, pezeshkpour2024optionorder, yang2025option, xue2024symbolbinding] —
order/symbol fragility and mitigations. [wang2024firsttoken] — logits vs generated text disagree.
[li2023labelsupervised, humeau2020polyencoders] — decoder-with-head and cached-context scoring.

**Genuinely new.** The *negative-and-positive pair* that pins the mechanism at the interface level: a
slot-agnostic semantic scorer over cached candidate embeddings (N2) loses 13 points of MMLU-Pro
among-K, while scoring the terminal state against each option's own contextual spans (N3) matches the
letter readout on Δ_q (N1 mean .109, N3 .106, teacher .102 over two seeds) with IIA Δ log-odds .09–.13
vs .44–.46 and reorder Δp .10 vs .17. That the readout must carry *slot identity* is an
implementation claim nobody else has had to make, because nobody else shipped the readout as the
output interface.

**Reproduction / confirmation.** "Hidden option states contain the answer" is [wong2026models]. The
paper must not claim this; NOVELTY.md already flags it.

**How a reviewer attacks it.** (i) The equivalence claim N3 ≈ N1 sits at the noise floor (seed spread
.01–.02 on Δ_q, and the claimed gap is ~.003). Equivalence needs more seeds or a TOST-style argument.
(ii) Order fragility is reduced but not eliminated — the permutation-invariant energy head gets
exactly 0, so the honest claim is "between letters and a set-invariant scorer". (iii) Both arms train
with per-step option shuffling, so part of the robustness is augmentation, not architecture; a
reviewer will ask for the no-shuffle ablation. **Verdict: contribution**, provided it is framed as an
interface/systems result (readout + invariance + cost), not as a mechanistic discovery.

## F3. The evidence-vs-knowledge trade-off was two artifacts: depth and a rendered abstain line

**Evidence.** §3r (tap 20 recovers evidence at equal Δ_q), §3t (removing the rendered "none of the
above" line fixes the null at every K at no knowledge cost), §3n–§3o and §3u (the two-expert
machinery it makes unnecessary, including a learned gate that recovers 0.9 of a 12.5-point envelope).

**Closest prior work.** [gromov2024deeperlayers, men2024shortgpt, schwartz2020righttool,
tenney2019pipeline, belrose2023tunedlens] — mid-depth features beat last-layer features for
classification. [zhao2021calibrate, tam2024speakfreely, robinson2023mcp] — rendering/format effects on
LM decisions. [larson2019clinc] — literal vs modelled out-of-scope.

**Genuinely new.** The composite diagnosis in a decision-head setting, plus its architectural
consequence: a proposed two-regime system (energy path for evidence, native path for knowledge)
collapses into one model once the tap and the rendering are fixed, and a learned confidence-feature
gate provably cannot recover the remaining envelope.

**Reproduction / confirmation.** The depth half is a confirmation — "the last layer is specialised for
next-token prediction" is textbook. Label it as such.

**How a reviewer attacks it.** (i) One tap value (71%) chosen post hoc and then ported across
backbones without a per-size sweep. (ii) "Artifact" is a strong word for "we changed a flag and it got
better" when several cells are single-seed. (iii) The null-line result is partly a property of our
training mix (only 0.9% of workflow rows carry a null target). **Verdict: contribution (narrow)** as a
negative-result/ablation section; not a headline claim.

## F4. Rubric-conditioned workflow training transfers within its rule grammar and not beyond

**Evidence.** §3w/§3x (full-row re-evaluation: held-out rubric styles +34, held-out families +9–14,
rubric-flip both-correct .43 → .57; every untouched external set at or below its constant-prediction
floor except tree-choice; two rubric-dependence probes that disagree), §3ad (level-7 composition ~.50
for every model).

**Closest prior work.** [lake2018scan, keysers2020cfq, kim2020cogs, hupkes2020compositionality] —
learned grammars do not compose out of distribution. [wei2022flan, sanh2022t0, chung2024flant5] —
what instruction tuning does generalise. [zhou2023lima, lin2024urial] — tuning mostly teaches format.

**Genuinely new.** Almost nothing at the level of mechanism. What is defensible is *evaluation
methodology*: floor-relative reporting (constant-prediction baselines per external file), rubric-flip
and rubric-shuffle probes, and the honest report that the two probes disagree.

**Reproduction / confirmation.** Entirely a confirmation in a new task family.

**How a reviewer attacks it.** (i) "Your generator wrote the training and the held-out families, so
in-grammar transfer is memorisation of your own DSL." (ii) The external suite is small, and several
files are at their majority floor for *both* checkpoints, which measures the benchmark, not the model.
(iii) No baseline of a frozen instruction-tuned LLM given the same rubric — without it, "not beyond"
is unquantified. **Verdict: not a contribution as a finding**; keep it as an evaluation-protocol
section and a negative result that motivates the curriculum work.

## F5. Fine-tuning lifts standard-tier accuracy and costs hard-tier accuracy versus a frozen backbone with few-shot prompting; calibration collapses without soft targets and calibration-based selection

**Evidence.** §3ab (frozen 14B 3-shot hard .559 vs trained 14B .468, standard .819 → .875; held-out
score NLL 2.87, the worst of the ladder), §3ah (long-state fix + `--best_on` calibration
selection — no Brier term trained: standard .931, score NLL 2.87 → 0.95, hard Brier .85 → .66, hard accuracy .450 — still
below frozen), §3ac (ordinal-smoothed targets: NLL 2.07 → 1.23 with identical argmax), §3ag (Qwen3.5
generation shifts hard-tier frozen performance).

**Closest prior work.** [kumar2022finetunedistort] — fine-tuning distorts pretrained features and
underperforms OOD; this is the single most dangerous citation for us, because it predicts our result.
[mosbach2023fewshot] — FT vs ICL comparisons. [biderman2024loralearnsless, luo2025forgetting] —
LoRA/forgetting trade-offs. [wang2025cogcalib, desai2020calibration] — fine-tuning-induced
miscalibration. [mueller2019labelsmoothing, mukhoti2020focalcalibration, gneiting2007scoringrules] —
soft targets and proper scoring rules as the fix.

**Genuinely new.** The quantified crossover on an external typed-decision benchmark at three model
scales, and — more useful — the isolation that the binding constraint was the *selection criterion and
target distribution*, not capacity: the largest single-run swing in the entire project came from
selecting checkpoints on held-out calibration NLL, not from any architecture change (no run trains a
Brier term; `--brier_lambda` is 0.0 throughout).
The inversion at Qwen3.5-4B (hard .495, above every Qwen3 trained model at any size) is a second
useful datum: pretraining generation, not parameter count, predicts hard-tier headroom.

**Reproduction / confirmation.** Both halves are known phenomena; the contribution is instantiation
plus the selection-criterion isolation.

**How a reviewer attacks it.** (i) Single seed per ladder point, and the 8B point is a self-declared
outlier — so "monotone in scale" is asserted on three points with one exception. (ii) The hard tier is
n = 111 public items; .468 vs .450 is inside noise, and the standard tier's n_eff is 36 states ×
2 paraphrases. (iii) Frozen-with-three-shots is not matched on inference cost, prompt search, or
rendering — and §3ag shows rendering alone moves the frozen number by 12 points, which is the same
order as the effect being claimed. (iv) The trivial explanation — our training distribution simply
does not cover the hard families — is not excluded. **Verdict: contribution (narrow)**; the
calibration-objective/selection half is the defensible part, the "training hurts hard tier" half must
be reported with the rendering confound stated in the same paragraph.

## F6. A data-truncation bug (facts rendered after long policies, right-truncated at the training window) produced confidently-wrong long-context behaviour

**Evidence.** §3ag (98.8% of long rows lost their facts at a 1,024-token window, 100% at 256),
§3ab (long_policy .05 at 14B, .42 at the untrained 1.7B base), §3ah (facts-first + `--drop_truncated`:
long_policy .05 → .158, multi_hop .44 → .611, level-7 .620).

**Closest prior work.** [liu2024lostmiddle, levy2024moretokens, shi2023distracted, hsieh2024ruler] —
inference-time position and length effects. [sambasivan2021datacascades] — silent data failures
compounding downstream.

**Genuinely new.** The framing (render position × fixed training window is invisible to a
length ablation) is a real practitioner's point and, as far as we found, unstated in this form.

**Reproduction / confirmation.** The underlying phenomenon is a bug, not a discovery.

**How a reviewer attacks it.** Fatally, on confounding: the fix was shipped inside `tl1b` together
with frozen-teacher KD, calibration-based checkpoint selection, 3,072-token states and
8k steps. No matched single-variable ablation (facts-first vs facts-last at an identical window and
seed) exists; `tl1b_nokd` isolates only the KD term. The quantitative claim "this caused the
long-context collapse" is therefore unsupported as run. **Verdict: not a contribution as stated.**
Either (a) demote to a reproducibility/practice note with the confound declared, or (b) run one
matched pair (facts-first vs facts-last, same window, same seed, ~$10) and promote it to a proper
ablation. Recommendation: (b) — it is the cheapest upgrade in this list.

## F7. Serving cost is dominated by launch overhead, not FLOPs

**Evidence.** §3ab (apples-to-apples single-pod: one decision 45 / 56 / 60 ms at 1.7B / 4B / 14B, an
8× parameter increase for 1.33× latency; marginal batched cost scales with option tokens, not K per
se), §3m (at M = 1 every path is fixed-cost dominated), §3ag (removing a KV-cache deep copy: 21–27 →
15.5–17 ms; CUDA graphs rejected for cross-bucket probability drift).

**Closest prior work.** [pope2023scaling] — inference regimes and what actually binds;
[kwon2023pagedattention, juravsky2024hydragen, zheng2023sglang, gim2023promptcache] — prefix reuse;
[leviathan2023speculative, chen2023speculativesampling, cai2024medusa] — the whole premise of
speculative decoding is that a decode step is not compute-bound.

**Genuinely new.** Nothing mechanistic. The numbers are useful as a systems observation for this
architecture class: with a cached state prefix and no decode loop, a decision costs roughly one
short-suffix prefill, so latency is sublinear in model size and a 14B decision model is a viable
product where a 14B generator is not.

**How a reviewer attacks it.** "Your overhead is your implementation's overhead" — single stream,
Python-level chunking, no comparison against an optimised runtime (vLLM/TensorRT-LLM) or a saturated
server, and the one attempt at graph capture was abandoned. **Verdict: not a contribution as a
finding**; keep as a measured engineering section supporting the product claim, with the caveat in
the text.

---

## Overall

**Strongest:** F1 (candidate-blind factorisation, with the shuffled-question control and the
full-depth closure) and F2 (direct contextual readout as the output interface, with slot identity as
the necessary ingredient and order fragility quantified). F3 supports both as the ablation that
removes the two confounds, and F5's calibration half is a genuinely useful training recipe.
**Weakest:** F6 and F7 are, as run, a bug report and an engineering measurement — neither is
defensible as a research finding without the matched ablation (F6) or an optimised-runtime baseline
(F7); F4 is a textbook compositional-generalisation confirmation whose only novel content is the
floor-relative evaluation protocol. The paper's title and abstract should be built on F1+F2, with
F3/F5 as supporting sections and F4/F6/F7 demoted to analysis and practice notes.
