# PLAN3 — Phase 3: knowledge retention under the PCDM cost constraint (2026-09-19)

Inputs: the PI's memo (2026-09-19, "be stricter"), RESEARCH_KNOWLEDGE.md (literature), RESEARCH_KNOWLEDGE_adversarial.md,
COMPARE.md, REPORT.md §3d–§3f. Budget: ~$35 of the ~$70 left.

## The constraint (protect this)
Expensive compute = C_state + M·C_query + O(K) GEMMs — never M·K·transformer. Some O(K) is unavoidable (K probabilities);
it must be dot products / tiny projections. Consequences: the "options in the suffix" pointer readout is a *teacher* and a
small-K diagnostic (`mcq_lora`), not the deployed architecture; listwise self-attention is O(K²) and already cost 4–7 pts
at large K.

## Thesis to test
PCDM compiles the *listwise*, candidate-conditioned reasoning of a pretrained LM into a reusable decision state
Z = g(state, query) computed without seeing candidate identities; arbitrary runtime candidates are introduced after Z exists
and scored by cheap late interaction s_j = F(Z', φ(a_j)), where Z' = Z plus at most one O(K) set-conditioning step (the LM
never sees candidates; a pure candidate-independent Z restores IIA and cannot represent a listwise teacher);
abstention is modeled as support of the answer set, not as one more softmax class. Metric that makes this a science result:
**Parametric Knowledge Retention = Acc_PCDM / Acc_same-backbone-teacher** on MMLU-Pro, under the evidence suite held within
seed floor and the K=1000 latency shape preserved.

Novelty is the combination + the compilation problem, not the parts (KV prefix reuse, late interaction, dynamic label
embeddings/GILE, split transformers/DeFormer/PreTTR/LUMEN, cascades, distillation, energy scoring are all prior art;
cross-encoder → bi-encoder distillation (Margin-MSE / TAS-B), ColBERT, MixEncoder ("Once is Enough", EMNLP 2023 — query encoded once, light parallel candidate interaction), ERNIE-Search and GILE are the closest cousins and must be cited and distinguished:
our target is answer reasoning + calibrated reject over runtime candidate sets, not relevance).

## Experiments (pre-registered)

| # | experiment | code change | $ | pass / kill |
|---|---|---|---|---|
| E0 | teacher gap on MMLU-Pro 1,200: `B_1.7B` (cloze log-prob), `B_8B`, `mcq_lora` (options in suffix, letter readout) vs `joint_emb*` (bi-encoder head, 9%) | none — running | 0.5 | decomposes the 9% into backbone ceiling / interface / training signal. If `mcq_lora` ≤ 15%: the 1.7B same-backbone teacher is too weak; E3 uses 8B teacher labels and reports retention vs both |
| E1 | depth under joint: `--tap_layer 28` and `[h20;h28]` (multi-tap concat), same candidate head, data v5, `mmlu_pro` added to eval | `--tap_layer 28`; `--tap_layers 20,28` concat (d_in doubles) | 8 | adopt if MMLU-Pro ≥ 15% and evidence suite within seed floor (NLI ±.7, trained large-K ±3, unseen ±5). Hypothesis: h20 = evidence alignment, h28 = answer formation |
| E2 | factorized null on the E1 winner: r = σ(g(max_j s, s(1)−s(2), mean s, logsumexp s − log K, cand-set mean/var, h)); P(∅)=r, P(a_j)=(1−r)·softmax(s)_j; loss = BCE(r, gold-absent) + CE on present rows | `model.py` (`--null factored`), `decision_loss` | 5 | K-sweep range ≤ .10 (now .43–.67), OOS recall ≥ .80, near-miss AUROC ≥ .95, large-K within ±3, IIA stays exact |
| E3 (revised 2026-09-19 after the PI's second memo) | **compile a listwise teacher into a reusable decision state.** E3-0 closed-book corpus (MMLU-aux, ARC, OBQA, CSQA, SciQ, QASC, LogiQA, MathQA/AQuA, MedMCQA subset; domains held out; never MMLU/MMLU-Pro). E3-T teacher = `mcq_lora` init + corpus hard labels → soft labels over the corpus; 8B 5-shot letters on MMLU-Pro as the size ceiling. E3-a/b/c students from the same teacher: Z₁ single vector · Z_R (R = 8 probes) + token-level MaxSim · Z_R + one cross-attention from the probes over all candidate tokens (O(KRd), the LM never sees candidates); loss α·CE + β·T²·KL (β = architecture-transfer, not truth); candidate encoder = tiny 2-layer over Qwen token embeddings (0 layers ⇒ table only); Qwen3-Emb stays the warm label-space tier. E3-cf counterfactual option sets on MMLU-Pro (remove 3 / add unrelated / replace hardest / add near-dup / reorder; paraphrase if an LLM key is provided), teacher distribution per variant, same Z once per question. E3-lat warm and cold latency per candidate encoder, no Python loops (flattened tokens + segmented reductions). | `data.py`/`scripts/distill_corpus.py`, `scripts/teacher_label.py`, `model.py` heads + cand encoder, `train.py --distill`, `scripts/mmlu_counterfactual.py`, `bench.py --cold` | 21 | **Knowledge Retention** = among-K_student / among-K_teacher ≥ 0.8 and **Abstention Error** reported separately (never conflate with acc); evidence suite within seed floor; NLL/ECE/Brier/ChaosNLI/null no worse than `joint_emb_lw_v5` beyond noise; cold ≤ 1.5× warm at K = 10, warm flat in K; the a/b/c ladder + counterfactual KL is the result (how much compiles into Z; how much O(K) set-conditioning is required). Knowledge Retention denominator = the 1.7B option-aware teacher's among-K (the question is compilability, not whether 1.7B knows enough); the 8B 5-shot number is a scale reference, never the denominator. All three students start from the base checkpoint (no nesting); a `zr → zr_set` continuation, if run, is reported as a separate "+set refinement". This run labels ONE option set per row (ordinary listwise KD); **E3-ms** (funded 2026-09-19; primary criterion = question-dependent gain (normal − choices-only among-K) > 0 on MMLU-Pro, since the choices-only control showed single-set KD compiled no question-dependent knowledge; secondary: counterfactual kl_teacher ↓, delta_mae ↓, `zr_set` > steps-matched control): expand each corpus row into M variant sets (remove / add-unrelated / reorder), teacher-label each, sum the KL — same-Z, multiple-set supervision that forces candidate-conditioned reasoning to factor through a candidate-blind decision state. Before reading any MMLU-Pro gain as generalization: exact + normalized + near-dup audit of data_kb against the frozen 1,200 items, and cross-source dedup inside data_kb (MMLU-aux bundles ARC/OBQA/RACE). |
| E4 | sparse residual: candidate-conditioned pass on fixed top-r (r = 8) → correction Δs, distilled back into the fast path | only if E3 retention < 0.8 | 8 | closes ≥ half the remaining gap at fixed r-cost |

Order: E0 → E1 → E2 → E3 (E1/E2 share a pod day). Not now: DeepSets set-context (only if E2 leaves set-dependence
necessary), KDA, MoE, latent iteration, 4B (only as a retention-ratio check after E3 works at 1.7B).

## Reporting
Every run reports the evidence suite (REPORT §3c columns), MMLU-Pro acc/among-K/ECE by category, K-sweep range, and the
bench marginal at K = 4/150/1000. Results go to REPORT.md §3g+ and the run registry in PROJECT.md.
