# PCDM — project report (v0 → v1), 2026-09-16/17

Consolidated record of what was built, what broke, what was learned, and where the hypotheses of `idea.md` stand.
Numbers: `runs/*/results.json` (temperature-scaled on val). Design docs: `PLAN.md` (v0), `PLAN2.md` (v1 + revisions).
Live logs: W&B project `guy-na8/pcdm`; artifacts: HF `guychuk/pcdm-runs` (private), data `guychuk/pcdm-data` (private).

## 1. What was built

| piece | v0 (Mac, MPS) | v1 (RunPod H100 ×2) |
|---|---|---|
| backbone | frozen Qwen3-0.6B-Base, layer 20, features cached per string | Qwen3-1.7B-Base, truncated at layer 20, LoRA r16 on layers 13–20 (hand-rolled), layers 1–12 frozen |
| state↔query interaction | 2-layer 512-d cross-attn tower (query tokens + slot → state tokens) | same tower; **plus joint prefix encoding** (`--joint`: query as causal suffix of the state, KV-cached at inference) |
| candidates | mean-pooled frozen features of the label text, cached | same, from frozen layer 12; **hybrid** frozen-space similarity features in the scorer |
| null | extra softmax logit from the slot (F) → candidate-aware (G) | candidate-aware (G) default |
| features | – | per-dim z-score of backbone features (`--zscore`) |
| data | 44k rows, 5 sources | 758k rows, 24 tasks, 21 eval sets incl. 3 fully held-out label spaces (Banking77, TREC, 20NG), 10 query templates/family, label-wording augmentation, leak audit |
| training | 8 epochs on cached features (minutes) | 12k–24k steps × 64, resumable checkpoints, W&B, HF upload, two-pod schedule scripts |
| baselines | prompted 0.6B log-probs; frozen cross-encoder + linear heads; pooled MLP | KV-cached prompted 1.7B and 8B; LoRA cross-encoder (`C_lora`) |
| tests | – | `tests/test_pipeline.py` (16 tests, CPU, 5 s): ragged masks, permutation equivariance, loss = hand soft-CE, LoRA grads, checkpoint/resume reproduces loss, joint causal invariance, KV-cache decide == concatenation |

Cost: v0 $0 (local); v1 ≈ $35 of H100 time for 16 runs (of the $145 balance).

## 2. Bugs and findings, in the order they bit

1. **Attention-sink token (v0).** Qwen3 has no BOS; position 0's hidden state is content-independent (cos 0.9998 between
   `neutral`, `yes`, `transfer`). Every single-token candidate was one vector. Fix: prepend `<|endoftext|>`, drop it.
2. **Fixed classifier in disguise (v0).** Three fixed NLI label strings → the scorer memorised three vectors; paraphrased
   labels scored 11% (below chance). Fix: randomise label wordings in training; keep 5 wordings held out for eval.
3. **State-only null leaks its base rate (v0).** `s_∅ = f(state)` learned the 13% synthetic-null prior and put it on every
   clean item (ChaosNLI NLL 1.33 vs 1.17 without null). Fix: candidate-aware null (variant G): NLL 1.15, ECE 0.034.
4. **Last-layer tap = hypothesis-only model (v1, the big one).** v1 read the backbone's *last* layer; SNLI 58.6 = the
   hypothesis-only baseline; the tower never used the premise. Tap layer 20 → 65.3 (frozen) / 68.2 (LoRA); NLI family
   loss finally descends. Backbone size was never the bottleneck (0.6B→1.7B bought nothing until the tap moved).
5. **Rogue dimensions (v1).** 3 dims carry ~30% of token norm (mean token cosine 0.45–0.64). Per-dim z-score:
   +3.5 SNLI / −0.08 NLL locally; +2–4 on CLINC/HWU64, best null AUROC at scale.
6. **Null over-fires on unfamiliar wordings — a data bug (v1).** Gold-absent synthesis used `K = N−1` near-miss sets
   (teaches "almost the whole label set ⇒ none") and a 25% rate. Data v3: 15%, K∈{3,10}. TREC/20NG accuracies doubled.
7. **LoRA dropout active at eval** (modules built after `from_pretrained`'s `eval()`); fixed.
8. **The cross-encoder is not the "expensive joint" alternative (v1, the reframing).** `C_lora` is a *causal* decoder over
   `premise [SEP] hypothesis`: premise states never see the hypothesis. It is itself late-interaction — its "state memory"
   is the premise KV cache and its "tower" is 28 pretrained layers applied to the hypothesis. Ours replaced those with 2
   fresh 512-d layers. Literature (DeFormer, PreTTR, LUMEN, Poly-encoders, ColBERT): keep pretrained joint layers over a
   precomputed prefix. Implemented as `--joint` with a bit-exact causal-invariance test. (Result: §4, pending.)
9. **Ops (RunPod).** `tmux kill-session` doesn't kill `timeout`-wrapped schedule children (a ghost schedule trained a second
   model on the same GPU); `pkill -f` over SSH kills the SSH shell; `--env` vars don't reach SSH sessions; two pods sharing
   one uv `.venv` on a volume corrupt it (`uv run` auto-syncs; use `--no-sync` + per-pod envs); driver 570 hosts need a
   cu126 torch wheel. All recorded in memory for next time.

## 3. Results (v1, temperature-scaled; single seed unless noted)

Runs: `main_s0` original v1 config (last-layer tap); `diag_tap20_frozen` / `diag_tap20` tap-20 diagnostics (no LoRA / LoRA);
`main_v2` tap-20 + LoRA + z-score; `main_v3` same + data v3, 2 epochs; `main_v3_s1` seed 1 (1 epoch); `abl_nohybrid` no
frozen-space similarity; `B_1.7B` / `B_8B` prompted LMs (3-shot, KV-cached candidate log-probs, literal "none of the above");
`C_lora` LoRA cross-encoder with per-task heads (no null, fixed label sets).

| metric | main_s0 | diag_tap20_frozen | diag_tap20 | main_v2 | main_v3 | main_v3_s1 | abl_nohybrid | B_1.7B | B_8B | C_lora |
|---|---|---|---|---|---|---|---|---|---|---|
| snli_test/acc | 0.586 | 0.653 | 0.682 | 0.681 | 0.701 | 0.685 | 0.692 | 0.791 | 0.818 | 0.855 |
| mnli_val/acc | 0.514 | 0.574 | 0.615 | 0.635 | 0.648 | 0.621 | 0.633 | 0.702 | 0.823 | 0.768 |
| anli_test/acc | 0.356 | 0.362 | 0.414 | 0.422 | 0.435 | 0.424 | 0.425 | 0.445 | 0.504 | 0.376 |
| boolq_val/acc | 0.492 | 0.592 | 0.628 | 0.628 | 0.663 | 0.624 | 0.641 | 0.782 | 0.863 | 0.598 |
| clinc_test/acc (K=150) | 0.562 | 0.551 | 0.611 | 0.654 | 0.708 | 0.673 | 0.669 | 0.450 | 0.387 | 0.393 |
| hwu64_test/acc (K=64) | 0.697 | 0.693 | 0.750 | 0.790 | 0.836 | 0.816 | 0.848 | 0.030 | 0.020 | 0.623 |
| clinc_heldout/acc (unseen intents) | 0.612 | 0.671 | 0.667 | 0.688 | 0.689 | 0.658 | 0.494 | 0.530 | 0.390 | – |
| clinc_heldout/acc_k | 0.761 | 0.729 | 0.764 | 0.769 | 0.771 | 0.729 | 0.704 | 0.830 | 0.875 | – |
| banking77_test/acc (K=77, held-out dataset) | 0.269 | 0.336 | 0.309 | 0.329 | 0.326 | 0.340 | 0.082 | 0.270 | 0.200 | – |
| banking77_k/acc (K=10) | 0.450 | 0.506 | 0.492 | 0.524 | 0.474 | 0.484 | 0.327 | 0.620 | 0.594 | – |
| trec_fine/acc (K=50, held-out) | 0.110 | 0.068 | 0.112 | 0.112 | 0.112 | 0.188 | 0.066 | 0.052 | 0.040 | – |
| ng20_test/acc (K=20, held-out) | 0.125 | 0.135 | 0.110 | 0.087 | 0.171 | 0.181 | 0.165 | 0.162 | 0.059 | – |
| snli_test_paraphrase/acc (unseen label wordings) | 0.120 | 0.269 | 0.329 | 0.376 | 0.398 | 0.375 | 0.439 | 0.080 | 0.016 | 0.841 |
| snli_test_hyponly/acc (premise blanked) | – | – | – | – | 0.648 | 0.652 | 0.615 | 0.346 | 0.387 | – |
| clinc_k/auroc_null | 0.858 | 0.858 | 0.876 | 0.897 | 0.919 | 0.901 | 0.902 | 0.645 | 0.617 | – |
| snli_null/auroc_null | 0.721 | 0.818 | 0.852 | 0.852 | 0.861 | 0.853 | 0.857 | 0.860 | 0.887 | – |
| squad_null/auroc_null | 0.749 | 0.780 | 0.842 | 0.855 | 0.882 | 0.858 | 0.829 | 0.831 | 0.886 | – |
| chaos_mnli/nll ↓ | 1.102 | 1.138 | 1.141 | 1.132 | 1.162 | 1.189 | 1.171 | 1.158 | 1.236 | 1.492 |
| snli_test/ece ↓ | 0.029 | 0.023 | 0.030 | 0.024 | 0.031 | 0.022 | 0.033 | 0.393 | 0.494 | 0.036 |
| fitted T | 0.96 | 0.96 | 1.04 | 0.96 | 1.12 | 1.12 | 1.12 | 3.63 | 4.58 | 0.96 |

Noise floor — clean seed pair `joint_v1` vs `joint_v1_s1` (identical config): NLI/BoolQ ±0.7 pt, CLINC/HWU64 ±1,
null AUROC ±0.014, ChaosNLI NLL ±0.05; but held-out label-space sets swing ±3–5 pts (banking77_k .411/.454, clinc_heldout
.647/.692) and paraphrased labels ±8 (.619/.542). Claims on those sets need ≥ 5 pts or two seeds. `C_lora`'s paraphrase cell is by
construction (it maps label strings to head indices). `C_lora` is 3k steps × 32 — a lower bound on the cross-encoder.

**H2 (latency, H100, `main_v3` vs KV-cached prompted 1.7B, one 256-token state):** M=256 queries at K=4: ours 0.18 s
(2.9× the M=1 cost; 0.62 ms/query marginal) vs 11.0 s (43 ms/query) → 69× cheaper; K=32: 35×; K=150: 19×. State
encode is a flat 20 ms. Pending: rerun on `joint_v1` (queries pass through 20 layers against the KV cache).

## 4. Hypothesis scorecard (pre-registered rules in PLAN.md / PLAN2.md)

| H | rule | status |
|---|---|---|
| **H1** accuracy retention | ≥ cross-encoder − 2 on SNLI/MNLI; ≥ 0.9× prompted 8B on ANLI | **Fails for the tower family**: 70/65 vs 86/77 (C_lora), 82/82 (8B); ANLI 43.5 vs 45.4 needed. Wins where the label space is large or runtime-defined: CLINC 71 vs 39/39, HWU64 84 vs 62/2, BoolQ 66 vs 60 (C_lora). **`joint_v1` pending** — val NLL 0.56 at 3k steps vs 0.65 best-ever for the tower; §2.8. |
| **H2** parallel cost | latency(M) ≪ M·latency(1) | **Holds**: 19–69× cheaper per query at M=256, growing with M. |
| **H3** probability quality | beat C_lora and 8B on ChaosNLI NLL and SNLI ECE | **Holds**: NLL 1.16 vs 1.49 / 1.24; ECE 0.03 vs 0.036 / 0.49. Prompted LMs need T≈4 and are still poorly calibrated. |
| **H4** null / unseen labels | AUROC > 0.85 on 3 null sets; held-out intents ≥ 0.8 | **Null: holds** (0.92 / 0.86 / 0.88). **Unseen labels: partial** — held-out CLINC intents 69% (8B: 39% acc / 88% among-K), Banking77 33% (8B: 20%), TREC/20NG < 20% for everyone. The hybrid frozen-space similarity is the whole mechanism (`abl_nohybrid`: Banking77 33 → 8). |

Net: the *economics, calibration, typed output and null* half of the thesis is demonstrated; *accuracy retention* on
pairwise NLI is not, for the fresh-tower design — and the diagnosis (§2.8) says why and what to change.

### 3b. Phase 4 — R1 `joint_v1` (prefix-conditioned query encoding), final

| metric | joint_v1 | main_v3 (tower) | B_1.7B | B_8B | C_lora |
|---|---|---|---|---|---|
| snli / mnli / anli / boolq acc | **0.910 / 0.874 / 0.544 / 0.815** | 0.701 / 0.648 / 0.435 / 0.663 | 0.791 / 0.702 / 0.445 / 0.782 | 0.818 / 0.823 / 0.504 / 0.863 | 0.855 / 0.768 / 0.376 / 0.598 |
| clinc K=150 / hwu64 / clinc-heldout / banking77 | 0.688 / 0.816 / 0.647 / 0.306 | **0.708 / 0.836 / 0.689 / 0.326** | 0.450 / 0.030 / 0.530 / 0.270 | 0.387 / 0.020 / 0.390 / 0.200 | 0.393 / 0.623 / – / – |
| paraphrased labels / hyp-only probe | **0.619** / 0.440 | 0.398 / 0.648 | 0.080 / 0.346 | 0.016 / 0.387 | (0.841) / – |
| null AUROC clinc / snli / squad | 0.902 / **0.979 / 0.971** | **0.919** / 0.861 / 0.882 | .645/.860/.831 | .617/.887/.886 | – |
| snli_soft NLL / unli NLL / chaos NLL | **0.558 / 0.530** / 1.264 | 0.814 / 0.613 / **1.162** | 1.016/.699/1.158 | 1.165/.698/1.236 | 0.642/.594/1.492 |
| SNLI ECE (T) | **0.019** (1.12) | 0.031 (1.12) | 0.393 (3.6) | 0.494 (4.6) | 0.036 (0.96) |

Reading: moving the state↔query interaction into the pretrained layers (query as causal suffix over the cached state) lifts
NLI by +21/+23 pts, above both the fine-tuned cross-encoder and the prompted 8B, with the best calibration and null detection
measured; the hypothesis-only probe drops to 44% (the model now reads the state; +47 pts from the premise vs +3–8 before).
Costs: ~2–4 pts on large-K classification and unseen-intent sets, and worse ChaosNLI NLL (sharper on ambiguous items) —
listwise candidates (IDEA2 §6) and Stage-B calibration training are the natural next steps for exactly those.

**The "fancy classifier over Qwen" concern (raised 2026-09-17).** With joint encoding, Qwen does the semantic alignment;
the decision head is the *interface*. Its measured value over the two "let Qwen do everything" alternatives: runtime-defined
label spaces (unseen intents 65% vs 39% prompted-8B; paraphrased labels 62% vs 2%), large K (HWU64 82% vs 2%), an explicit
null (AUROC .97 vs .89), calibration (ECE .02 vs .49), and one K-independent suffix pass per query. `abl_notower` (queued)
tests whether the cross-attention slot itself still matters after joint encoding; if not, it should be removed.

## 3c. Phase 2 (after the stop-and-rethink review; data v4 = K-decoupled nulls + choice-set/K-sweep/null-slice eval sets)

All no-tower joint models; `abl_notower`/`mcq_lora` on data v3 (paired), `joint_v2`/`joint_lw`/`joint_emb` on data v4 (paired).

| metric | ours (abl_notower, v3) | **MCQ-LoRA** (v3) | zs 8B letters | joint_v2 (v4) | joint_lw (v4, listwise) | **joint_emb** (v4, embedder cands) |
|---|---|---|---|---|---|---|
| snli / mnli / anli / boolq | **.908 / .877 / .547 / .818** | .853 / .727 / .400 / .768 | .34 / .36 / .34 / .61 | .905 / .873 / .524 / .818 | .906 / .867 / .523 / .822 | .906 / .873 / .524 / .817 |
| clinc-150 / hwu64 | .784 / .862 | .715 / .580 | .00 / .02 | .777 / .862 | .709 / .834 | **.803 / .875** |
| clinc-heldout acc / among-K | .612 / .800 | **.890 / .920** | .12 / .12 | .507 / .785 | .615 / .790 | .614 / .855 |
| banking77 K=77 acc | .303 | **.568** | .01 | .286 | .309 | .440 |
| banking77 K=10 acc / among-K | .473 / .613 | **.637 / .830** | .08 / .14 | .552 / .634 | .524 / .653 | .566 / .754 |
| trec-coarse / trec-fine / ng20 among-K | .34 / .20 / .24 | **.62 / .43 / .53** | .17 / .04 / .16 | .35 / .19 / .24 | .30 / .20 / .22 | .38 / .22 / .31 |
| paraphrased labels acc | .547 | .808 | .34 | .639 | .565 | **.810** |
| clinc-OOS null recall | .657 | .388 | .74 | .723 | **.854** | .649 |
| null AUROC clinc-k / snli / squad | .946 / .979 / .979 | .676 / .960 / .992 | .51 / .50 / .74 | .950 / .979 / .980 | .952 / .980 / .978 | .953 / .979 / .976 |
| ChaosNLI NLL / SNLI ECE | 1.248 / .020 | **1.126** / .028 | 1.25 / .026 | 1.356 / .007 | 1.394 / .010 | 1.381 / **.005** |
| P(∅ | gold absent), K=2 → K=150 (range) | – | – | – | .98 → .35 (.63) | .93 → .56 (**.37**) | .97 → .31 (.67) |
| add-irrelevant Δlog-odds (IIA) | 0 | – | – | 0.000 | 0.144 | 0.000 |

**Verdicts.**
- *Decision head vs "Qwen does everything" (MCQ-LoRA, same everything but the readout):* the energy head wins in-distribution
  (+5.5 SNLI, +15 MNLI, +15 ANLI, +5 BoolQ, +7 CLINC-150, +28 HWU64), on null detection (CLINC AUROC .95 vs .68) and on cost
  (K-independent; MCQ training steps ~4× slower, two-stage chunking above 51 options). **MCQ-LoRA wins on unseen label spaces**
  (+28 CLINC-heldout, +27 Banking77-77, TREC/20NG ≈ 2×, +26 paraphrased labels) because the option strings are read in context
  with pretrained label semantics. The head is not decorative; it trades label-space generality for accuracy, calibration, null
  and cost.
- *Candidate encoder (`joint_emb`):* replacing mean-pooled decoder features by an embedding model is the best single change:
  best in-distribution numbers, paraphrased labels 81% (= MCQ), Banking77-77 44% (from 29%), ECE .005. The remaining gap to
  MCQ on unseen spaces is mostly **false null** (CLINC-heldout among-K .855 vs acc .614).
- *Listwise (`joint_lw`):* reduces the structural K-dilution of the null (range .63 → .37; OOS recall 85%) but not to the ≤ .10
  target, costs ~3–7 pts on large-K sets, and breaks IIA (as designed). H5: partial.
- *H2, fair (`bench_fair`, prefix-sharing baseline):* per-query marginal at M=256: ours 0.7 / 1.5 / 4.4 / 25.5 ms vs B_fair
  53 / 56 / 87 / 416 ms at K = 4 / 32 / 150 / 1000 → **73× / 37× / 20× / 16×**. Sharing the prefix barely helps the log-prob
  baseline because its cost is K candidate continuations, not the state.
- *Zero-shot 8B with enumerated letters* is at chance (a base model without instruction tuning cannot use the letter format);
  the log-prob `B_8B` remains the prompted reference.

## 3d. Follow-ups on `joint_emb` (2026-09-18, ~$10; `joint_emb_lw`, `joint_emb_s1` on pod 3)

**(1) K-aware null bias, post-hoc (`--dump_logits` + `scripts/null_bias.py`, $0 after a 15-min eval dump).**
Fit `s∅' = s∅ + α·log K + β` jointly with T on val by soft-CE, apply to every eval set without retraining.
Result: **the val fit is a no-op** — α = 0.00, β = −0.25, T = 0.96 (val soft-CE 0.3897 → 0.3888). In-distribution the model
has already learned to compensate for softmax dilution; the K-sweep range is unchanged (.67 → .70). Held-out sets move
±1–2 pts (CLINC-heldout .612 → .627, TREC .208 → .232), CLINC-OOS −6.

Cross-OOD transfer probe (fit on one unseen label space, apply to the others) shows *why* no scalar can close the false-null gap:

| fit on | β | CLINC-heldout | TREC-coarse | Banking77-K | CLINC-OOS |
|---|---|---|---|---|---|
| val (in-dist.) | −0.25 | .627 | .232 | .552 | .590 |
| Banking77-K (has gold-absent rows) | **+2.0** | .461 | .008 | .635 | .886 |
| TREC-coarse (gold always present) | **−6.0** | **.850** | .376 | .377 | .001 |

The fitted offset flips sign with the fitting set. CLINC-heldout's gap (.612 vs .853 among-K) closes only by effectively
switching the null off (β = −6 → .850), which kills OOS detection (.001). **Verdict: the false-null gap on unseen label
spaces is a novelty/absence confound, not a K-bias** — unfamiliar label *vocabulary* looks like "gold absent" (null AUROC
.745 on Banking77-K vs .952 on trained CLINC vocabulary). Fix must come from training signal that separates the two
(gold-present rows over held-out label vocabularies, i.e. label-space-level held-out splits in training), not from calibration.
`runs/joint_emb_nullbias/results.json` holds the val-fitted variant in train.py's schema.

**(2) `joint_emb + listwise` (`joint_emb_lw`, $4).** Same as `joint_emb` plus the `SetMixer` (identity-at-init) over
candidates. Best val NLL of any run (0.381 vs 0.389).

| | joint_emb | **joint_emb_lw** | Δ |
|---|---|---|---|
| SNLI / MNLI / ANLI / BoolQ | .906 / .873 / .524 / .817 | .905 / .873 / .521 / .815 | 0 |
| CLINC-150 / HWU64 | .803 / .875 | .763 / .869 | **−4** / −1 (listwise large-K cost; was −7 on layer-12 candidates) |
| CLINC-heldout acc / among-K | .614 / .855 | **.696** / .866 | **+8** / +1 → false-null gap .24 → .17 |
| Banking77-77 / Banking77-K / TREC-coarse / 20NG | .440 / .566 / .210 / .273 | .436 / .551 / .296 / .295 | ≈ (+9 TREC-coarse, no Banking77 loss) |
| paraphrased labels | .810 | .820 | +1 |
| CLINC-OOS null recall | .649 | **.823** | **+17** |
| P(∅ \| absent) K=2 → K=150 | .97 → .31 (range .67) | .91 → .48 (range **.43**) | dilution halved, still > .10 target |
| null AUROC clinc-k / irrq-intent / near-miss | .953 / .939 / .972 | .960 / .952 / .967 | ≈ |
| ChaosNLI NLL / UNLI NLL / SNLI ECE | 1.381 / .554 / .005 | **1.323 / .531** / .006 | better on disputed items |
| IIA add-irrelevant Δlog-odds / dup mass err | 0.000 / .047 | 0.168 / .046 | IIA broken by design; duplicates handled the same |

Verdict (after the seed pair in (3)): the effects of listwise that clear 2× seed noise are **null-side** — OOS recall
+17/+22, P(∅|absent) range .67/.70 → .43 — and its cost, −4/−7 on CLINC-150 (the one large-K trained space). The apparent
unseen-label gain (CLINC-heldout +8 vs seed 0) is +3 vs seed 1, i.e. within the ±5 seed spread; TREC/20NG/paraphrase shifts
likewise. No NLI or Banking77 cost. IIA broken by design (Δlog-odds 0.17). H5 stays partial (range .43 > .10). Reading
consistent with §3d(1): the set view helps *whether to abstain* (a coherent unfamiliar set vs one distractor), but does not
close the novelty ⇒ null confound. **Default for null-critical use: `joint_emb_lw`; for large-K accuracy: `joint_emb`.**

**(3) second `joint_emb` seed (`joint_emb_s1`, $4).** Noise floor for every claim below:

| | seed 0 | seed 1 | |Δ| |
|---|---|---|---|
| SNLI / MNLI / ANLI / BoolQ | .906 / .873 / .524 / .817 | .908 / .874 / .531 / .817 | ≤ .007 |
| CLINC-150 / HWU64 | .803 / .875 | .829 / .869 | .026 / .007 |
| CLINC-heldout acc / among-K | .614 / .855 | .665 / .866 | .051 / .011 |
| Banking77-77 / Banking77-K / TREC-coarse / 20NG | .440 / .566 / .210 / .273 | .470 / .561 / .246 / .314 | .03 / .00 / .04 / .04 |
| paraphrased labels | .810 | .836 | .026 |
| CLINC-OOS recall / null AUROC clinc-k | .649 / .953 | .598 / .959 | .051 / .007 |
| ChaosNLI NLL / SNLI ECE / val NLL | 1.381 / .005 / .389 | 1.359 / .009 / .384 | .02 / .004 / .005 |
| P(∅ \| absent) range | .665 | .701 | .036 |

Rule of thumb from both seed pairs (`joint_v1`, `joint_emb`): NLI ±0.7, trained large-K ±3, unseen label spaces and OOS
±5. Any single-seed difference under those is noise; this demotes several earlier "+2–4" readings in §3c to ties.

## 3e. Data v5 — label-vocabulary diversity (`joint_emb_lw_v5`, 2026-09-18, A100 ≈ $5)

Test of the §3d verdict ("the false-null gap on unseen label spaces needs training signal, not calibration"). **One variable
changed**: training classification data = 30 label vocabularies × ≤ 4,000 rows (14 new sources: bbc-news, subj, CR, enron-spam,
amazon-counterfactual, sst2, tweet-stance, arxiv-11, patent-9, bitext-support-27, student-subjects, hate/offensive, imdb,
yelp-5) instead of v4's 16 vocabularies × up to 30,000 rows. NLI/BoolQ/SQuAD/CLINC untouched; **eval files byte-identical to
v4** (v5's eval dir overwritten with v4's); same config as `joint_emb_lw`, same seed, 12k steps.

| | joint_emb_lw (v4) | **joint_emb_lw_v5** | Δ | seed floor |
|---|---|---|---|---|
| Banking77 K=77 | .436 | **.527** | **+9** | ±3 |
| Banking77-K acc / among-K | .551 / .757 | **.606 / .817** | +5.5 / +6 | ±1 |
| TREC-fine / TREC-coarse | .180 / .296 | **.308 / .356** | **+13** / +6 | ±4 |
| 20NG among-K | .295 | .336 | +4 | ±4 |
| CLINC-heldout acc / among-K | .696 / .866 | **.762** / .876 | +7 / +1 | ±5 |
| Banking77-K null AUROC | .752 | **.804** | +5 | ±0.5 |
| SNLI / MNLI / ANLI / BoolQ | .905 / .873 / .521 / .815 | .909 / .880 / **.555** / **.834** | +.4 / +.7 / **+3.4** / +1.9 | ±.7 |
| CLINC-150 / HWU64 | .763 / .869 | .784 / .818 | +2 / **−5** (HWU64 train rows 9k → 4k) | ±3 |
| paraphrased labels | .820 | .837 | +2 | ±3 |
| CLINC-OOS recall / P(∅ \| absent) range | .823 / .43 | .735 / .52 | **−9** / worse | ±5 |
| null AUROC clinc-k / irrq-intent / near-miss-hwu64 | .960 / .952 / .967 | .961 / .971 / .942 | ≈ / +2 / −2.5 | |
| ChaosNLI NLL / SNLI ECE (raw → T) | 1.323 / .003 → .006 | **1.273** / .009 → .020 | T = 1.22 fitted on v5's classification-heavier val over-softens NLI | |

**Verdict: the unseen-label gap is mostly data.** Every unseen-vocabulary metric moved 1.5–3× the seed floor in the same
direction, and the novelty ⇒ null confound shrank (Banking77-K null AUROC .75 → .80). The gap to MCQ-LoRA narrowed from
~25 pts to 4–20: Banking77-77 .527 vs .568, TREC-fine .308 vs .430, 20NG .336 vs .534, CLINC-heldout .762 vs .890. Bonus:
ANLI +3.4 and BoolQ +1.9 from *classification* diversity — the shared LoRA was over-fitting a few intent vocabularies.
Costs: the capped in-distribution intent set (HWU64 −5) and the null side (OOS recall −9, K-dilution range .43 → .52) — the
model now abstains less on unfamiliar-looking sets, which is exactly the trade §3d predicted.
Next knobs are data-build costs, not model changes: more vocabularies (60+), lower cap (1–2k), and a per-family T (or
per-task T) instead of one global T.

## 3f. MMLU-Pro — the one number TypeSafe/Jev publishes (2026-09-19, $0.3)

`scripts/mmlu_pro_eval.py`: 1,200 items, stratified over 14 categories (seed 0), 10 options (K = 10 for 986 items), gold
always present, T fitted on v4 val. Jev: 84.6% (third-party 1,200-item sample, ECE .031). Ours:

| | acc | among-K | ECE | NLL |
|---|---|---|---|---|
| joint_emb | .082 | .132 | .33 | 3.39 |
| joint_emb_lw | .090 | .130 | .31 | 3.32 |
| joint_emb_lw_v5 | .092 | .123 | .39 | 3.85 |
| random (K = 10) | .10 | .10 | – | 2.30 |
| `B_8B` cloze log-prob (option as continuation, other options not visible; 600 items) | .185 | .225 | .08 | 2.31 |
| **`mcq_lora`** (same 1.7B, options rendered in the suffix, letter readout, evidence-only training) | **.235** | **.307** | .19 | 2.41 |
| `B_1.7B` cloze (600 items) | 0.115 | 0.143 | 0.05 | 2.37 |

**E0 decomposition (PLAN3):** the interface is the first-order cause. `mcq_lora` — identical backbone and training data, zero
knowledge MCQ rows — recovers .235/.307 because the LM *reads the options against each other*; the bi-encoder head cannot
(9%). The 8B cloze baseline, which scores each option blind to the others, sits *below* the 1.7B options-in-context readout —
so the distillation teacher for E3 must see the options, and the same-backbone retention target is ≈ .30 (Qwen3-1.7B
direct-answer with options in context). Depth (20 vs 28 layers) is still confounded in this table; E1 separates it.

**Chance, and confidently wrong.** Three stacked reasons: (1) nothing in training is parametric-knowledge QA — every task is
"the answer is in the state", so the head learned evidence ↔ candidate fit, which is meaningless for "3.2 × 10⁵ J" vs
"the court lacked jurisdiction"; (2) Qwen3-1.7B-Base *in full* is ~30–35% on MMLU-Pro and we tap layer 20 of 28, cutting
the layers where answer formation happens; (3) the novelty ⇒ null confound (§3d) fires — among-K > acc means ~30% of items
are abstained on because the option strings look like an unfamiliar vocabulary. Reading: PCDM as built is a
decision-over-evidence model (NLI .91, BoolQ .83, 150-way intents .80 at supervised-model level); Jev is a general-knowledge
System-One model. This benchmark measures the latter and we never set out to have it — a different class of model, not a
worse version. What would make the comparison apples-to-apples: Jev on our evidence-grounded eval files (needs API access),
and the prompted `B_8B` log-prob baseline on the same 1,200 items to separate "backbone knowledge" from "our head" (queued).

## 3g. PLAN3 E1 — depth under joint encoding (`tap28_v5`, 2026-09-19, $5) — **killed**

Same as `joint_emb_lw_v5` (tap 20) but the suffix runs all 28 layers with LoRA on 21–28. Pre-registered rule: adopt if
MMLU-Pro ≥ 15% and the evidence suite stays within seed floor.

| | tap 20 (`joint_emb_lw_v5`) | tap 28 |
|---|---|---|
| SNLI / MNLI / ANLI / BoolQ | .909 / .880 / .555 / .834 | .880 / **.809** / **.468** / **.707** |
| CLINC-150 / HWU64 / CLINC-heldout / Banking77-77 | .784 / .818 / .762 / .527 | .775 / .796 / .731 / .490 |
| CLINC-OOS recall / null AUROC clinc-k | .735 / .961 | .667 / .951 |
| MMLU-Pro acc / among-K / ECE | .092 / .123 / .39 | .091 / .117 / .28 |
| best val NLL | .371 | .452 |

Depth is not the lever: last-layer states hurt evidence alignment (the v1 finding survives joint encoding) and add nothing
on knowledge through the bi-encoder interface. `[h20;h28]` was cancelled (h28 alone adds no MMLU signal, so the concat
cannot). Combined with E0 (§3f): the +17 from `mcq_lora` is the *interface* (options read in context), not depth or size.
E3 (compile the option-conditioned teacher into a candidate-independent Z) is the remaining lever within the cost constraint.

## 3h. PLAN3 E2 — factorized null (`factnull_v5`, 2026-09-19, $5) — **killed as a K-dilution fix; kept as a finding**

P(∅) = r(set statistics, h), P(a_j) = (1−r)·softmax(s)_j, T on candidate scores only; same config/data/seed as
`joint_emb_lw_v5`. Rule: K-sweep range ≤ .10, OOS ≥ .80, near-miss AUROC ≥ .95, large-K within ±3.

| | softmax null (`joint_emb_lw_v5`) | factored |
|---|---|---|
| NLI / BoolQ | .909 / .880 / .555 / .834 | .908 / .879 / .552 / .828 |
| CLINC-150 / HWU64 | .784 / .818 | .765 / .798 |
| CLINC-heldout acc / among-K | .762 / .876 | .717 / .875 (more abstention, same discrimination) |
| Banking77-77 / Banking77-K / TREC-fine / 20NG | .527 / .606 / .308 / .336 | .490 / .639 / .222 / .293 |
| CLINC-OOS recall | .735 | **.780** |
| null AUROC clinc-k / irrq / near-miss hwu64 / clinc | .961 / .971 / .942 / .906 | .958 / .977 / .935 / .894 |
| P(∅ \| absent) K=2 → 150 (range) | .94 → .41 (.52) | .93 → .45 (.48) |
| IIA Δlog-odds / dup mass err / val NLL | .243 / .030 / .371 | .121 / .030 / **.365** |

**Why it cannot work, and what the K-sweep actually measures.** Null AUROC *by K* on the sweep is .99 / .96 / .90 / .85 at
K = 2 / 10 / 50 / 150 for every model we have (softmax, factored, listwise, pointwise). The null loses *discrimination*
with K, not calibration: at K = 150 "gold absent" leaves the gold's sibling intents in the set, so absence is a near-miss
judgment the head cannot make — the sweep conflates K with difficulty. Modeling abstention as answer-set support does not
change that (the gate's max-score statistic legitimately rises with K), and any consistent probabilistic model has P(∅|absent)
falling with K when near-misses carry mass. Consequences: (1) the "range ≤ .10" target is mis-specified; the right target is
null AUROC at *fixed difficulty* — a sibling-excluded K-sweep (`ksweep_*_nosib`) is added to the eval; (2) the lever for
large-K abstention is fine-grained discrimination (E3's option-conditioned teacher), not the null's functional form;
(3) factored null stays available (`--null factored`) — +4.5 OOS, best val NLL — but is not the default.

**Sibling-excluded K-sweep (`ksweep_clinc_nosib`, `joint_emb_lw_v5`):** P(∅|absent) / null AUROC at K = 2 / 50 / 150 =
.95/.99 → .69/.94 → .59/.90 without siblings vs .94/.99 → .56/.90 → .41/.85 with. So about half of the large-K drop was
near-miss difficulty and half is genuine K-dependence — an extreme-value effect over 149 unrelated distractors that costs
9 AUROC points on its own. The correct statement of "K-dilution" is therefore: discrimination degrades with K even at fixed
difficulty, and near-misses roughly double the effect; neither the softmax form nor set-support modeling changes it.

## 3i. Fair latency, corrected (`bench_a100`, A100 80GB PCIe, 2026-09-19)

Two artifacts fixed in `decide` (per-query embedder calls → one batched call; per-candidate Python lookups → one gather).
Baseline `B_batched` = log-prob scoring with the state prefix shared **and queries batched in chunks of 32** (explicit dense
causal+pad mask; matches sequential `B_fair` to 2e-5). Per-query marginal at M = 256, warm candidate vocabulary:

| K | ours | B_batched | B_fair (sequential) | ours vs batched | peak mem ours / batched |
|---|---|---|---|---|---|
| 4 | 1.29 ms | 4.3 ms | 50.7 ms | **3.3×** | 8.8 / 13.6 GB |
| 32 | 1.27 | 20.3 | 57.8 | **16×** | 8.8 / 25.9 |
| 150 | 1.34 | 92.1 | 141.8 | **69×** | 9.1 / 25.4 |
| 1000 | 2.44 | 650.8 | 731.0 | **267×** | 14.6 / 46.0 |

Retires the earlier "73× at K = 4" (that baseline did not batch across queries). The right statement is L(K) = C_LM + α·N_candidate-tokens with small α — expensive LM compute independent of K, then cheap
approximately linear candidate work (1.3 ms to K = 150, 2.4 ms at K = 1000; the O(K·d) scorer term PLAN3 factorizes in E3) — not "flat in K". **Caveat (PI memo):** these are
*warm* numbers — candidate vectors precomputed. For runtime-novel candidates (MMLU-style) the candidate encoder is on the
request path; E3-lat measures cold vs warm per candidate encoder.

## 3j. PLAN3 E3 — compiling a listwise teacher into a decision state (2026-09-19; closed — negative, see the choices-only control)

**Corpus (`data_kb`, `scripts/distill_corpus.py`):** 147,446 rows from MMLU-aux (40k), AQuA (40k), MedMCQA (20k), LogiQA
(12.6k), SciQ (11.7k), CSQA (9.7k), QASC (8.1k), OBQA (5k), ARC (3.4k); val 3,001; TruthfulQA-MC1 held out (817); GPQA
gated. **Audit (`scripts/kb_audit.py`, `runs/kb_audit.json`):** 127,180 unique normalized stems (duplication is
within-source: MMLU-aux 9.1k groups, AQuA 2.5k; cross-source ≤ 18 pairs) — the corpus is ~127k independent items, not
147k. Leakage vs the frozen 1,200 MMLU-Pro items: exact 1, normalized 1–2, MinHash@0.8 1, stem+options 0 → **2 items**
(qids 8011, 8375; `data_kb/leaked_qids.json`), excluded from reported MMLU-Pro numbers.

**Teacher (`teacher_kb`):** `mcq_lora` warm start, 4k steps on data_kb, options rendered in the suffix, letter readout.
MMLU-Pro acc .294 / among-K **.310** / ECE .10 (vs `mcq_lora` .235 / .307): the corpus mostly removed abstention;
knowledge is at the 1.7B ceiling. TruthfulQA-MC1 .321 / .425. Knowledge Retention denominator = .310 (8B 5-shot = scale
reference only). This run labels ONE option set per row (ordinary listwise KD); same-Z multi-set supervision (E3-ms,
`scripts/multiset.py`) is the follow-up if the ladder is positive.

**Teacher choice-set behaviour (`mmlu_cf`):** orig .312 among-K; remove-3 .376 among-K but acc **.073** (the null
letter shifts with K — abstention error .30); add-3-unrelated .273 (−4); replace-hardest .307; near-dup .399; reorder
Δp .097; IIA Δlog-odds **.179**. The listwise teacher is choice-set-fragile.

**E3b — `zr` student (R = 8 probes, token MaxSim, tiny cold-capable candidate encoder, single-set KD, $5.5):**

| | baseline `joint_emb_lw_v5` | `e3b_zr` | teacher |
|---|---|---|---|
| MMLU-Pro among-K / acc / ECE | .123 / .092 / .385 | **.157 / .157 / .039** | .310 / .294 / .100 |
| **KR_raw** (among-K / teacher among-K) | 0.40 | **0.51** | 1.00 |
| **KR_excess** ((among-K − chance .10) / (teacher − chance)) | 0.11 | **0.27** | 1.00 |
| Abstention error (among-K − acc) | .031 | **.000** | .016 |
| TruthfulQA-MC1 among-K / kb val among-K | – | .286 / .344 | .425 / – |
| counterfactual orig / remove3 / add3 / replace / near-dup / reorder | – | .158 / .230 / .113 / .153 / .112 / .157 | .312 / .376 / .273 / .307 / .399 / .312 |
| IIA Δlog-odds / reorder Δp | 0 / 0 | **0 / 0** (exact) | .179 / .097 |
| SNLI / MNLI / ANLI / BoolQ | .909 / .880 / .555 / .834 | .905 / .873 / .541 / .826 (within noise) | – |
| CLINC-150 / HWU64 | .784 / .818 | .724 / .784 | – |
| CLINC-heldout / Banking77-77 / TREC-fine / 20NG | .762 / .527 / .308 / .336 | **.391 / .097 / .138 / .261** | – |
| CLINC-OOS / null AUROC clinc-k / ChaosNLI | .735 / .961 / 1.273 | .568 / .909 / 1.313 | – |

Readings. (1) E3 more than doubles the above-chance teacher capability captured by a candidate-blind Z under ordinary
single-set KD — KR_excess .11 → .27 (raw .40 → .51; "half the knowledge compiles" is the wrong reading, 10-way chance is .10) — with MMLU calibration improved (ECE .39 → .04) and no abstention error; the student is exactly
set-stable where the teacher is fragile. (2) `add3_unrel` costs the student 4.5 pts under exact IIA, so the residual is
discrimination (unrelated options sometimes outscore the gold), not set-dependence per se — `zr_set` decides whether
O(K) set-conditioning recovers it. (3) **Unseen label spaces collapsed** (Banking77-77 .527 → .097): the tiny candidate
encoder (Qwen embedding table + 2 layers from scratch) carries no pretrained label semantics; all label-space transfer in
this project came from Qwen3-Embedding (§5a). The PI's two-tier structure is therefore required, not optional — semantic
embedder for the label-space tier, tiny encoder for cold runtime candidates. `e3d_zr_emb` (same head with the semantic
embedder) is queued to confirm the head keeps label transfer. Evidence tasks otherwise held within seed noise (CLINC-150
−6, HWU64 −3 are the same-encoder effect on trained label spaces).

**E3c — `zr_set` (zr + one O(K) cross-attention from the probes over all candidate tokens; same data, seed, steps; $5.5):**
MMLU-Pro among-K .147 vs `zr` .157 (KR_excess .23 vs .27; within the ±1-pt SE of 1,200 items), counterfactuals track `zr`
(remove3 .216 / add3 .107 / near-dup .096), learned IIA Δlog-odds .034, reorder exact. Evidence identical. Null side
better: CLINC-OOS .568 → .727, null AUROC .909 → .924, MMLU ECE .011, ChaosNLI 1.31 → 1.23. Reading: **under single-set
KD there is no supervision for set dependence, so the set-conditioning path cannot learn what the teacher's residual
r_j(Z, A) is** — the ladder's `zr ≈ zr_set` is the expected null result of single-set training, not evidence that the
residual is small. E3-ms (same-Z, multi-set, symmetrized teacher, Δ-log-odds targets) is the test that can separate them.

**E3a — `z1` (single decision vector, factorized bilinear scorer; $5.5) and the complete ladder:**

| MMLU-Pro | `z1` | `zr` | `zr_set` | teacher |
|---|---|---|---|---|
| among-K | **.171** | .157 | .147 | .310 |
| KR_raw / KR_excess | .55 / **.34** | .51 / .27 | .48 / .23 | 1 / 1 |
| ECE | .046 | .039 | .011 | .100 |
| counterfactual orig / remove3 / add3 / replace / near-dup / reorder | .172 / .237 / .142 / .182 / .131 / .170 | .158 / .230 / .113 / .153 / .112 / .157 | .147 / .216 / .107 / .138 / .096 / .149 | .312 / .376 / .273 / .307 / .399 / .312 |
| TruthfulQA among-K / kb val among-K | .267 / .352 | .286 / .344 | .274 / .346 | .425 / – |
| CLINC-OOS / null AUROC clinc-k | .493 / .929 | .568 / .909 | .727 / .924 | – |
| SNLI / MNLI / ANLI / BoolQ | .907 / .876 / .536 / .817 | .905 / .873 / .541 / .826 | .909 / .875 / .537 / .831 | – |

Ladder reading (PI frame, 2026-09-19): z1 ≈ zr ≈ zr_set ≈ .15–.17 ≪ teacher .31 — **neither decision-state capacity nor
set-dependence is the bottleneck under single-set KD**; the single vector retains as much as eight probes or probes + set
conditioning (differences within ~2 SE of n = 1,200). About a quarter to a third of the teacher's above-chance capability
compiles into a candidate-blind Z regardless of head form; the rest is either irreducible without option-conditioned LM
computation or needs (i) the same-Z multi-set + Δ-log-odds signal (E3-ms pilot) and/or (ii) a semantic candidate encoder
(`zr_emb`). Set-conditioning buys null quality only (OOS .49 → .57 → .73 across the ladder) — consistent with §3h: the
null benefits from seeing the set even when accuracy does not.

**Scale reference (`mcq8B_5shot_mmlu`, Qwen3-8B-Base, 5-shot letter prompt, $0.5):** among-K .287, acc .285, ECE .06;
reorder Δp .21, IIA Δlog-odds .21. Below the fine-tuned 1.7B teacher — few-shot letter prompting of a base model is a
weak, symbol-biased protocol, not a size ceiling. The proper scale reference is an 8B fine-tuned options-in-context teacher
(~$15; deferred). The 1.7B teacher's .310 stays the KR denominator.

**E3-lat — warm vs cold candidate latency (A100, ms/query at M = 256; cold = candidate strings arrive with the request,
nothing cached):**

| candidate encoder | K = 4 warm / cold | K = 32 | K = 150 | cold/warm |
|---|---|---|---|---|
| Qwen3-Embedding-0.6B (label-space tier, `joint_emb_lw`) | 1.65 / 2.23 | 1.74 / 2.33 | 2.52 / 3.77 | 1.35 / 1.34 / 1.50 |
| tiny (embedding table + 2 layers, `e3b_zr`) | **0.78 / 0.79** | 1.03 / 1.08 | 0.75 / 0.87 | **1.01 / 1.05 / 1.15** |

Both meet the ≤ 1.5× rule (the embedder tier exactly at the bar at K = 150). The tiny tier is essentially cold-free and
~2× cheaper warm, but carries no label semantics (§3j E3b); the embedder tier keeps label transfer and pays 35–50% for
novel strings. Two tiers, or the PI's follow-up — distill Qwen3-Embedding into the tiny encoder on a large phrase corpus —
are the two ways to get one encoder with both properties.

**Choices-only / shuffled-question control (`scripts/mmlu_probes.py`, PI memo; Balepur et al. ACL 2024) — decisive:**

| among-K | normal | choices only (question = ".") | shuffled question | question-dependent gain |
|---|---|---|---|---|
| teacher (1.7B, options in context) | .310 | .222 | .208 | **+.088** |
| `z1` student | .171 | .181 | .147 | −.010 |
| `zr` student | .157 | .166 | .182 | **−.009** |
| `zr_set` student | .147 | .162 | .151 | −.014 |

Every student's whole above-chance score is candidate-set priors (uniform across heads → the training signal, not the architecture) (option-text artifacts learned from the corpus); the
question contributes nothing. **Retraction:** §3j's "KR_excess .27 / a quarter to a third of the teacher's above-chance
capability compiles into Z" is wrong — under single-set KD, *no question-dependent knowledge compiled into Z*. The
teacher's own question-dependent component is small (+.088 among-K over its choices-only score), so the target E3 is
chasing at 1.7B is ~9 points, not ~21. Consequences: (1) every MMLU-Pro number in this section must be read as
(normal − choices-only); (2) the E3-ms pilot's primary criterion becomes question-dependent gain > 0 (with
counterfactual KL / Δ-reproduction secondary); (3) the honest framing of E3 so far: the decision-state factorization
transferred the teacher's *candidate priors and calibration* (ECE .39 → .04) but not its question-conditioned knowledge.

**E3-ms pilot (same-Z multi-set supervision; 2026-09-19, $3.5) — killed by its pre-registered rule.** 30k corpus rows ×
(orig + 6 variants: remove/add-unrelated/reorder = invariance; replace-hardest/near-dup/remove-strong = set-context),
teacher symmetrized over 3 permutations, Δ-log-odds targets over shared candidates; `zr_set` continued 4k steps from
`e3c_zr_set` with `CE + KD + γ·Huber(Δ^S − Δ^T)` vs a steps-matched single-set control from the same init.

| | control (single-set) | multi-set + Δ |
|---|---|---|
| MMLU-Pro among-K / acc / ECE | .124 / .124 / .035 | .130 / .098 / .071 |
| counterfactual kl_teacher (orig / add3 / replace / near-dup / reorder) | .48 / .48 / .46 / .46 / .48 | .56 / .54 / .55 / .52 / .56 |
| Δ-log-odds MAE (add3 / replace / near-dup / reorder) | .742 / .673 / .731 / .776 | .739 / .673 / .731 / .776 |
| near-dup among-K / CLINC-OOS / null AUROC | .089 / .701 / .930 | .046 / .572 / .918 |
| SNLI / MNLI / BoolQ | .913 / .881 / .831 | .911 / .881 / .833 |

No criterion moved in the right direction: teacher-KL got worse, Δ reproduction is *identical* (the O(K) set-conditioning
path learned no set dependence even with direct Δ supervision — its zero-initialised cross-attention stayed inert at γ = 1
over 4k steps), MMLU is noise, near-dup/OOS degraded. Δ_q probes: multi-set normal .130 / choices-only .113 / shuffled-question .130 (Δ_q = +.017 vs choices-only, **0.000 vs
shuffled**); control .124 / .147 / .147 (−.023). A wrong question helps the multi-set model exactly as much as the right one —
no question-dependent knowledge; the +.017 is a candidate-prior artifact of the choices-only rendering. The stricter
Δ_q(shuffled) = normal − shuffled is adopted as the primary measure from here. Verdict for the candidate-blind compilation branch at 1.7B: **no question-dependent knowledge compiles
into Z under single-set or multi-set distillation**; what transfers is candidate priors and calibration. The branch stops
here (PLAN4 §10, §19 Phase A); the mainline is native choice (§3k).

## 3k. Redirect — PLAN4 (2026-09-19): closed-set System-One decision model

The candidate-blind Z(x, q) constraint was stronger than the product/research goal. The E3 ladder + choices-only control
(§3j) showed that under single-set KD no question-dependent knowledge compiles into a candidate-blind state, while the same
backbone *with options in context* has it (Δ_q = +.088). PLAN4 redirects the mainline to P(y | x, q, A): the closed choice
set participates in one non-generative decision pass (state KV-cached, one suffix per query, no per-candidate LM call, no
answer generation), read out by a direct head over the terminal decision state h_D with candidate representations
(N2 semantic, N3 contextual, N2N3 hybrid) vs the letter-logit control N1. Large K routes through the existing energy scorer
→ top-r → native choice. E3-ms continues as the optional compilation branch. Success axes (PLAN4 §15): evidence retention,
Δ_q ≈ teacher's, decision quality (NLL/ECE/Brier/ChaosNLI/null, disaggregated abstention), one-suffix-per-query latency
over K = 2…256. `report_native.py` prints that table.

## 3l. PLAN4 Phase B/C — `native_choice_v1` (2026-09-19/20, H100)

`--readout native`: state prefix + query + lettered options + terminal decision token in ONE suffix through all 28 layers
(LoRA on 21–28); h_D = terminal state; N2 = factorized bilinear scorer over (h_D, Qwen3-Embedding candidate); factored null
over set statistics; options shuffled every step; training mix = data v5 + data_kb (hard labels), 12k steps, 0.72 s/step.

| | PCDM `joint_emb_lw_v5` | E3 `zr` | **`nc_n2`** | `nc_n1` (letter readout, same mix) | teacher (N1, kb only) |
|---|---|---|---|---|---|
| MMLU-Pro among-K / acc / ECE | .123 / .092 / .39 | .157 / .157 / .04 | **.198 / .143 / .16** | pending | .310 / .294 / .10 |
| TruthfulQA-MC1 among-K / kb val among-K | – | .286 / .344 | **.401 / .463** | | .425 / – |
| SNLI / MNLI / ANLI / BoolQ | .909 / .880 / .555 / .834 | .905 / .873 / .541 / .826 | **.854 / .742 / .410 / .775** | | (`mcq_lora`: .853 / .727 / .400 / .768) |
| CLINC-150 / HWU64 | .784 / .818 | .724 / .784 | .695 / .721 | | |
| CLINC-heldout / 20NG / TREC-fine / Banking77-77 | .762 / .336 / .308 / .527 | .391 / .261 / .138 / .097 | **.800** / .382 / .154 / **.012** | | |
| CLINC-OOS / null AUROC clinc-k / snli | .735 / .961 / .980 | .568 / .909 / .980 | .251 / .795 / .960 | | |
| P(∅ \| absent) K = 2 / 10 / 50 / 150 | .94 / .81 / .56 / .41 | – | .57 / .61 / .38 / **1.00** | | |
| counterfactual orig / remove3 / add3 / replace / near-dup / reorder | – | .158 / .230 / .113 / .153 / .112 / .157 | .198 / .237 / .146 / .182 / .167 / .196 | | .312 / .376 / .273 / .307 / .399 / .312 |
| IIA Δlog-odds / reorder Δp | 0 / 0 | 0 / 0 | .379 / .059 | | .179 / .097 |
| ChaosNLI NLL / SNLI NLL | 1.273 / .257 | 1.313 / .265 | **1.175** / .385 | | |

**`nc_n1` (letter readout, same mix) — the §16 comparison:** MMLU-Pro among-K **.325** / acc .273 / ECE .086 (above the kb-only
teacher's .310), TruthfulQA .465, kb val .547, CLINC-heldout **.918**, Banking77-77 .531, 20NG .541; but SNLI/MNLI/ANLI/BoolQ
.838 / .684 / .377 / .769, HWU64 .510, CLINC-OOS .633, null AUROC .826, IIA .46, reorder Δp .17, val NLL .512 (vs N2 .451).
**§16 rule: "N1 works, N2 fails → focus on the readout, not scale."** N1 *is* a readout over h_D — h_D·E[letter_j], where
the letter embedding carries slot identity that h_D encodes; a bilinear scorer against a slot-agnostic semantic candidate
cannot recover "the answer is slot C". N3 (h_D against each option's own contextual states, which carry slot identity) is the
direct fix and is queued as the last run in budget. Across the three axes no single model wins: knowledge N1 ≫ N2 > PCDM;
evidence PCDM ≫ N2 > N1; null PCDM ≫ N1 > N2 — empirical support for the two-regime architecture, and the evidence
regression of the options-in-suffix formulation is the open problem.

**Δ_q probes (choices-only / shuffled-question), PLAN4 §15B:**

| | normal | choices-only | shuffled-q | Δ_q | Δ_q / teacher |
|---|---|---|---|---|---|
| teacher (kb only) | .310 | .222 | .208 | .088 | 1.00 |
| `nc_n1` (letters) | .325 | .213 | .211 | **.112** | 1.27 |
| `nc_n2` (direct head) | .198 | .121 | .139 | **.077** | **0.87** |
| E3 `zr` (candidate-blind) | .157 | .166 | .182 | −.009 | ≈ 0 |

(Under the stricter shuffled-question variant adopted in §3j: teacher .102, N1 .114, N3 .116, N2 .058, v2 .074, v3 .103 —
same ordering, smaller ratios: N3 = 1.14× teacher, N2 = 0.57×.) The direct head N2 retains 57–87% of the teacher's
question-dependent knowledge depending on the variant; its lower raw score is mostly
*less candidate-prior exploitation* (choices-only .121 vs .213) — N2 is less artifact-driven, not less knowledgeable, with a
third of N1's order bias (reorder Δp .06 vs .17). This is the first non-generative readout in the project that exposes
question-conditioned parametric knowledge; the candidate-blind Z never did. N3 (contextual candidates) tests whether the
rest of the raw gap is recoverable. (`false_abstain` column in `report_native.py` needs verification — it disagrees with
among-K − acc; use the latter until fixed.)

**`nc_n3` (h_D scored against each option's own contextual hidden states, slot identity preserved; $9):**

| same mix / steps | MMLU-Pro among-K / acc / ECE | TruthfulQA / kb val | CLINC-heldout / Banking77-77 / 20NG / TREC-fine | IIA / reorder Δp | SNLI / MNLI / ANLI / BoolQ / CLINC-150 / HWU64 | null AUROC / OOS / P(∅\|absent) K=150 |
|---|---|---|---|---|---|---|
| `nc_n1` letters | .325 / .273 / .086 | .465 / .547 | .918 / .531 / .541 / .414 | .463 / .165 | .838 / .684 / .377 / .769 / .697 / .510 | .826 / .633 / .20 |
| `nc_n2` semantic | .198 / .143 / .162 | .401 / .463 | .800 / .012 / .382 / .154 | .379 / .059 | .854 / .742 / .410 / .775 / .695 / .721 | .795 / .251 / 1.00 |
| **`nc_n3` contextual** (Δ_q **.111** = 1.26× teacher; choices-only .203, shuffled .198) | **.314** / .133 / .459 | **.461 / .580** | **.916 / .497 / .559 / .400** | **.126** / .101 | **.863 / .752 / .431** / .737 / **.740** / .660 | .803 / .204 / .99 |
| PCDM energy | .123 / .092 / .385 | – | .762 / .527 / .336 / .308 | 0 / 0 | .909 / .880 / .555 / .834 / .784 / .818 | .961 / .735 / .41 |

**§16, first branch: N3 ≈ N1 → adopt direct native choice.** The mechanistic prediction held — the direct head needs
candidate representations that carry slot identity (the option's own contextual states), not slot-agnostic semantic
vectors; with them the non-generative readout matches the letter readout on every knowledge and unseen-label set, with a
quarter of its IIA fragility (.13 vs .46; one run — v2/v3 with the same readout give .25/.21) and evidence retention
between N1 and N2 (better on SNLI/MNLI/ANLI, worse on BoolQ .737 vs .769/.775 and HWU64 .660 vs .721). Established: *candidate-
conditioned pretrained reasoning can be exposed directly as typed probabilities without the generative answer interface.*
Two known defects, both in the null/calibration layer rather than the decision: (i) the native null does not transfer
across K — abstention error .18 on MMLU-Pro (acc .133 vs among-K .314), ECE .46, P(∅|absent) .99 at K = 150 while OOS
recall is .20; the set-statistics gate over h_D sees score distributions at K = 77–150 unlike the kb-heavy mix; (ii) the
evidence gap to the energy path (5–13 pts) persists for all native readouts — the options-in-suffix formulation, not the
readout, is the cause. Next (PLAN4 §16 "proceed to calibration training", §15 axes A and C): a K-robust native null
(train the null on the energy path's per-K null distribution, or route ∅ through the energy tier), and the evidence
regression (mixing ratio, evidence tasks rendered without letters, longer training). Budget exhausted at this point.

Readings on `nc_n2`: (1) knowledge moves — MMLU-Pro .198 (best non-letter readout so far),
TruthfulQA .401 ≈ teacher; (2) **evidence regresses 5–15 pts to `mcq_lora`'s level** — the options-in-suffix formulation
costs evidence accuracy at 1.7B/12k steps regardless of readout (PLAN4 §15A not met as trained); (3) **the native null
does not generalize across K** — abstains on everything at K = 150 (P(∅|absent) 1.00, Banking77-77 .012) while
under-firing on OOS (.25): the set-statistics gate over h_D sees score distributions at K = 77–150 unlike anything in the
kb-heavy mix; (4) set-dependent by design (IIA .38), modest order bias (Δp .06 with per-step shuffling).

## 3m. PLAN5 Phase 5A — native-choice latency with state-KV reuse (`bench_native`, H100, 2026-09-20)

`nc_n3` native path (state prefix cached once; per query ONE right-padded option-aware suffix, batched in chunks of 32,
h_D + option spans pooled, scored) vs the energy path (`joint_emb_lw`) vs the query-batched log-prob baseline; per-query
marginal ms at M = 256, real intent-label candidates.

| L_s = 256 | K = 2 | 4 | 10 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| suffix tokens | 31 | 43 | 79 | 233 | 470 | 1040 | 2320 |
| **native** | **1.07** | 1.14 | 1.55 | 3.38 | 6.44 | 17.3 | 43.8 |
| energy | 0.97 | 0.96 | 0.96 | 0.98 | 0.96 | 1.01 | 1.06 |
| batched log-prob | 2.12 | 2.77 | 4.59 | 12.2 | 22.8 | 49.8 | 101.7 |
| native / energy | 1.1× | 1.2× | 1.6× | 3.4× | 6.7× | 17× | 41× |

Longer states (rerun with native chunk 8 / baseline chunk 4, so small-K native marginals are less batched than above):

| | K = 2 | 4 | 10 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| L_s = 1000: native / energy / batched | 3.7 / 1.34 / 11.9 | 3.7 / 1.33 / 13.0 | 3.8 / 1.34 / 18.6 | 4.8 / 1.36 / 38.4 | 8.2 / 1.36 / 66.9 | 18.1 / 1.43 / 128.6 | 38.7 / 1.48 / 247.5 |
| L_s = 2000: native / energy / batched | 3.7 / 1.90 / 13.9 | 3.8 / 1.91 / 17.3 | 4.1 / 1.93 / 27.6 | 5.9 / 1.90 / 63.8 | OOM (baseline) | | |

Reading (corrected after review): at fixed chunk 8 native is *flat* in L_s (3.69 ms at 1k vs 3.66 at 2k for K = 2); the
1.07 → 3.7 ms jump from the L_s = 256 table is the chunk change (32 → 8), not the prefix. The energy path is the one that
grows with L_s (0.97 → 1.34 → 1.90 ms). The baseline at L_s ≥ 1k runs at chunk 4 with a per-sub-batch KV copy, so its
numbers here are inflated; the 2–3× statement holds only for the L_s = 256 run.

Single-decision and small-batch latency (L_s = 256; total ms per decision incl. the state pass):

| | M = 1 (one decision) | M = 32 (marginal ms) |
|---|---|---|
| native, K = 2 / 10 / 32 | **52 / 53 / 52** | 1.8 / 2.2 / 4.1 |
| energy, K = 2 / 10 / 32 | 83 / 84 / 85 | 2.8 / 2.8 / 2.9 |
| batched log-prob, K = 2 / 10 / 32 | 65 / 67 / 69 | 2.6 / 5.1 / 12.7 |

At M = 1 every path is fixed-cost dominated (state pass ≈ 20–23 ms + one forward) and **native is the fastest for
K ≤ 32 (≈ 52 ms on an H100)** — the energy path carries the embedder and its own overheads. The K-scaling above is a
throughput statement at M = 256; a service answering one bounded decision at a time is native-first regardless of K ≤ 32.

Caveats: the batched log-prob baseline deep-copies its KV cache per sub-batch (pre-existing; inflates it at large K —
the energy-vs-native comparison is the clean one); K = 256 candidates are 226 real intent names + 30 synthetic labels;
bench.json for this run was lost to the OOM (numbers transcribed from the log; the rerun checkpoints per row).

**Operating region (corrected).** The bench's own crossover field says: at M ≥ 32 the energy path is cheaper at *every* K
(K* = 2); native is cheaper only at M = 1, and there only because its fixed per-call overhead is lower (K* = 128 at M = 1).
So "K* ≈ 10–32" is a chosen cost ratio (native ≤ 1.6–3.4× energy), not a measured crossover. The honest statement: native
choice costs 10–60% more than the energy path per query up to K ≈ 10 and 3.4× at K = 32 in throughput mode, is the faster
path for single decisions, and scales linearly in option tokens; large sets go energy → top-r → native with r set by the
latency budget (PLAN4 §9). One suffix per query, no
generation, no per-candidate LM call — the systems requirement of PLAN4 §15D holds; latency scales with total option tokens,
not with K per se.

## 3n. PLAN5 Phase 5B — support gate × native choice (`scripts/compose_support.py`, eval-time, $1)

P(∅) = 1 − r with r = 1 − P_energy(∅) from `joint_emb_lw_v5`; P(a_j) = r · P_native(a_j | answerable) from `nc_n3`; no
retraining; aligned per row by candidate string over the shared eval sets.

| | energy | native | **composed** |
|---|---|---|---|
| MMLU-Pro among-K / acc / abstention error / ECE | .123 / .095 / .028 / .392 | .314 / .133 / .182 / .459 | **.314 / .222 / .092 / .204** |
| CLINC-OOS null recall | .739 | .194 | **.786** |
| null AUROC clinc-k / snli / squad | .961 / .980 / .975 | .803 / .964 / .992 | .961 / .980 / .975 |
| P(∅ \| absent) K = 2 / 50 / 150 | .94 / .59 / .45 | .86 / .40 / .99 | .94 / .59 / .45 |
| CLINC-heldout / Banking77-77 | .756 / .529 | **.916** / .497 | .794 / .483 |
| SNLI | .910 | .863 | .863 |

Reading: the composition inherits native's choice (among-K unchanged at .314) and the energy path's support signal
(OOS, AUROC, K-sweep = energy's). MMLU abstention error halves and ECE drops .46 → .20 — but not "for free": the
evidence-trained null still abstains on 9% of closed-book questions, and on unseen label spaces it imports the energy
path's novelty ⇒ null confound (CLINC-heldout .916 → .794). Verdict: keep the two-expert split for support vs choice;
the support gate needs either the v5-style label-diversity treatment or a rule that closed-book (no-evidence) questions are
not gated by the evidence null. `compose_n3_T` (energy T = 1.22 applied to r) is within noise of the T = 1 composition.
**With `nc_v2` (native null fixed, §3p):** composing the energy gate now *hurts* closed-book decisions (MMLU acc .282 → .216,
ECE .037 → .185) and buys only +9 OOS recall (.699 → .791). Rule adopted: the evidence-trained support gate applies to
evidence-grounded decisions only; closed-book abstention is the native head's own.

## 3o. PLAN5 Phase 5C — evidence/knowledge fusion at score level (`scripts/fuse_scores.py`, eval-time, $0)

Geometric mixture over shared candidates, s_j = s_j^E + g·s_j^N (energy null as support gate), swept over g on the
`joint_emb_lw_v5` + `nc_n3` dumps:

| g | MMLU-Pro among-K | SNLI / MNLI / ANLI / BoolQ | CLINC-150 | CLINC-heldout | Banking77-77 | OOS recall | SNLI ECE |
|---|---|---|---|---|---|---|---|
| 0 (= energy) | .123 | .910 / .881 / .554 / .835 | .782 | .756 | .529 | .739 | .009 |
| 0.5 | .190 | .908 / .878 / .539 / .835 | **.802** | .784 | **.563** | .732 | .037 |
| 1 | .224 | .902 / .866 / .527 / .822 | **.806** | .793 | **.576** | .730 | .053 |
| 2 | .253 | .893 / .846 / .507 / .803 | .799 | **.803** | .576 | .728 | .077 |
| native alone (`nc_n3`) | .314 | .863 / .752 / .431 / .737 | .740 | .916 | .497 | .194 | – |
| oracle (best g per set) | .253 | .910 / .881 / .554 / .835 | .806 | .803 | .578 | .739 | .009 |
| val-selected global g = 0.25 | .158 | .909 / .878 / .544 / .835 | .797 | .772 | .550 | .733 | .026 |

Linear mixture behaves the same (g = 0.5: MMLU .230, CLINC .795, Banking77 .565, OOS .785, SNLI .904).

Reading: on label-space decisions the two experts are **complementary** — at g ≈ 0.5–1 fusion beats both (CLINC-150 +2.4
over energy / +6.6 over native; Banking77-77 +5 / +8) for ≤ 1.5 pts of NLI. On closed-book knowledge a global g caps at
.25 vs native's .314, and on NLI any g > 0 costs. The oracle row (evidence sets at g = 0, label/knowledge sets at g ≥ 1)
is the target for a *per-input* gate; a val-selected global g captures little because val is evidence-heavy. Caveats (review): g is selected on the eval sets and the oracle row is a per-column max over the grid — an upper bound, not a
result; and the fused scores were not re-temperature-fitted, so probability quality degrades with g (Banking77 NLL 2.04 → 2.65,
ChaosNLI 1.27 → 1.97, ANLI ECE .17 → .30 from g = 0 to 2) — accuracy gains only until T is refit per g. Energy rows in §3n/§3o are
the T = 1 dump (MMLU acc .095 / ECE .392), not the T-fitted results.json (.092 / .385). Verdict for PLAN5 §4: not interference —
complementary *accuracy* signals with a task-dependent mixing weight; a learned per-input gate g(x, q) with a refit T is the
experiment, not a global mix.

## 3p. PLAN5 Phase 5A-v2 — letter-free native choice (`nc_v2`: `<choice>` tags, per-step shuffle, perm-consistency λ = 0.1; H100, $9)

| | `nc_n3` (letters) | **`nc_v2`** (tags) | goal |
|---|---|---|---|
| MMLU-Pro among-K / acc / abstention error / ECE | .314 / .133 / .181 / .459 | .286 / **.282** / **.004** / **.040** | keep knowledge: **failed** on Δ_q |
| Δ_q (normal − choices-only; shuffled) | .111 (.203; .198) | **.066** (.220; .212) | 0.75× teacher vs 1.26× |
| TruthfulQA / kb val | .461 / .580 | .477 / .538 | |
| reorder Δp / IIA Δlog-odds | .101 / .126 | .116 / .247 | Δp < .03: **failed** |
| SNLI / MNLI / ANLI / BoolQ / CLINC-150 / HWU64 | .863 / .752 / .431 / .737 / .740 / .660 | .863 / .756 / .441 / .727 / .749 / .663 | recover evidence: **unchanged** |
| CLINC-heldout / Banking77-77 / 20NG / TREC-fine | .916 / .497 / .559 / .400 | .903 / .357 / .536 / .334 | |
| CLINC-OOS / null AUROC clinc-k / P(∅\|absent) K = 150 | .204 / .803 / .99 | **.707 / .953 / .28** | (unplanned) native null fixed |

Readings. (1) **The native null is fixed by the rendering**: with no letter-rendered "none of the above" line, ∅ is purely the
head's decision over set statistics, and the K-pathology of §3l disappears (abstention error .18 → .004, OOS .20 → .71,
AUROC .80 → .95). But the question-dependent signal drops (Δ_q .111 → .066) while option-prior exploitation rises
(choices-only .203 → .220) — among-K hides this; only the probe shows it — for abstention this beats the support-gate composition (§3n) on closed-book questions,
though the composition still wins on evidence-grounded nulls — and the composition keeps N3's full Δ_q. (2) Letters were **not** the cause of
the evidence gap (unchanged) nor of the order sensitivity (Δp .10 → .12; perm-consistency at λ = 0.1 did nothing — with
letters gone, position is the only identity signal in the suffix, so the LM's positional bias survives). (3) Large tagged
suffixes hurt unseen label spaces at K = 77 (Banking77-77 −14). Net: `nc_v2` met none of its three goals; its null fix is real. **`nc_v3` (letters for options, no rendered ∅ line, no perm
loss; $9) isolates the two factors:** Δ_q **.108** (≈ N3's .111; v2 .066), among-K .287, acc .273 (abstention error .014),
ECE .069, null AUROC .940, P(∅|absent) at K = 150 .56, choices-only .178 (the least option-prior exploitation of any native
model), OOS .435 (v2 .707), evidence .858 / .740, reorder Δp .117, Banking77-77 .317. Reading with the review's caveats: v2 vs v3 differ in rendering *and* perm-loss (0.1 vs 0), v3 vs n3 drop the ∅ line
*and* "Answer:" — neither factor is isolated, and Δ_q .066 vs .108 is ~1.7 SE unpaired at one seed. What is supported:
removing the rendered ∅ line removes the MMLU over-abstention (abstention error .18 → .014); tags + perm-loss cost
.03–.04 Δ_q at one seed. What is *not* supported: "the null is fixed" — the K-pathology moved from K = 10 to K = 77:
Banking77-77 false-abstain is .36 (v2) and .52 (v3) vs .01 for n3, and v3's K-sweep is non-monotone (K100 .09, K150 .56).
`nc_v3` is the best native checkpoint on Δ_q + closed-book calibration, not on abstention generally.
Order sensitivity is untouched by any rendering (Δp .10–.12). Next levers: inference-time symmetrization
(average over 2–3 permutations; 2–3× native cost, still under the log-prob baseline), λ ≫ 0.1, and for the evidence gap the
formulation itself (mixing ratio / evidence rows rendered without options / longer training).

## 3q. JevBench (fstandhartinger/jevbench v1.2.1, public subset; 2026-09-20)

Harness: 231 public decisions (72 standard / 48 easy / 111 hard; the 146 judge items and all held-out items are not
public), typed `noul` (yes/no) / `choice` (K = 2–6) / `score` (ordinal); models return a probability distribution over the
exact label set; native distributions only (`pcdm_jev/`, `scripts/jevbench_run.py`; adversarially reviewed, defects fixed).
**This is a public-subset run — not a ranked entry** (the harness ranks only ≥ 95% coverage incl. judge); comparisons
below are per-item on the identical 231 ids from the harness's own `per-task.json`.

| accuracy on the same public ids | standard (72) | easy (48) | hard (111) | notes |
|---|---|---|---|---|
| **PCDM energy** `joint_emb_lw_v5` | .403 | .875 | .360 | Brier .75 / .20 / .84; p50 .19 s |
| **PCDM native** `nc_n3` | .417 | .854 | .306 | Brier .70 / .20 / .84; p50 .13 s |
| **PCDM native** `nc_v2` (letter-free, null fixed) | .472 | **.938** | .315 | Brier .64 / .18 / .77; p50 .13 s |
| **PCDM native** `nc_v3` (letters, no ∅ line) | .472 | .812 | .297 | |
| `nc_n3` with `noul` labels reversed (order control) | .458 | .854 | .351 | ±4 pts from label order alone |
| open-jev-deberta-v3-large (classifier, local CPU) | .431 | 1.00 | .378 | closest transport to ours |
| GLiNER2 / jeff (GLiFormer 400M) / Laya (ModernBERT) | .639 / .750 / .694 | 1.00 | .369 / .387 / .351 | small encoders trained on the task family |
| open-alternative-jev (Qwen3.5-4B) | .833 | 1.00 | .568 | |
| system-one-open (Gemma E2B LoRA) / system-one (Qwen3-8B) | .931 / – | 1.00 | .486 / .486 | |
| SemIf (Qwen3.5-4B) / OpenJev (26B-A4B) / djev | .986 / .972 / .986 | 1.00 | .613 / .640 / .676 | |
| Jev 1.13.0 (closed) | .986 | 1.00 | .730 | |

Disclosures (per the review): probabilities are the head's softmax **conditioned on non-∅** (P(∅) dropped; mean p_null
.20 energy / .03 native, share > .5: 14% of hard items for energy, 0% native); checkpoints trained at 256-token states /
64-token queries and run at 4096 / 256 (68% of hard states and 33–58% of energy queries exceed training length; none
truncated); latency is in-process on one H100, one decision at a time, model load excluded, cold label embedding included
(energy) — the harness's ×2 + 0.15 s self-hosted adjustment would apply, and their p50 is over standard+judge; cost is
null (no tariff); option order = harness label order (reversed-order control: standard .417 → .458, hard .306 → .351 for `nc_n3`); compose mode
≡ native under the harness and is not reported; harness commit, checkpoint sha256 and `uv.lock` are in each run's manifest.

Reading: majority/chance baselines are .311 standard / .284 easy / .336 hard — **PCDM's hard-tier numbers are at chance**,
standard is 1.5–3 SE above chance (n = 72, SE .058), and the reversed-label swing (±4) is larger than the differences
between our own models, so no ranking among n3/v2/v3 is supported. PCDM lands with the untrained classifiers
(DeBERTa-large .431/.378), not with systems trained on workflow decisions (.83–.99 standard). JevBench shifts the leading hypothesis from architecture to training distribution: PCDM performs well on
easy bounded decisions but degrades sharply on harder rubric-conditioned workflow judgments, the family Jev is explicitly
optimized for (PLAN6). It does not *prove* the gap is task family; live alternatives: state length (native hard .444 on the 36 states ≤ 256 tokens vs
.240 on the 75 longer ones; energy .36 / .36), `noul` at K = 2 where native is at chance (.46–.50) while energy gets .62,
and the ∅-conditioning (14% of energy hard items had p_null > .5 and were forced to answer). The strongest evidence *for*
task family over size is on the leaderboard itself: a Gemma-E2B LoRA reaches .931 standard. The
architecture side is competitive (p50 .13–.19 s raw vs .17–.24 s for the GPU entries, native distributions, K up to 6
trivially). `nc_v2`'s better calibration lifts easy to .938 and standard to .472 with no task training. JevBench is
therefore the target task family for Phase 6/7 (typed primitives + rubric-conditioned decision data + calibration), not a
benchmark to tune on; the 72 MIT-licensed original items are the only public training-eligible material and are too few.

## 3r. The depth confound resolved — native N3 at tap 20 (`nc_n3_tap20`, H100, 2026-09-20, $9)

The review of §3l–§3q flagged that every candidate-blind student was trained at tap 20 and every native model at 28 layers,
so "the native formulation costs evidence" (§3l) and "candidate-blind Z cannot compile knowledge" (§3j) were confounded
with depth. Same N3 readout, same v5 + kb mix, same steps, `--tap_layer 20`:

| | energy `joint_emb_lw_v5` | native N3 @ 28 (`nc_n3`) | **native N3 @ 20** |
|---|---|---|---|
| SNLI / MNLI / ANLI / BoolQ | .909 / .880 / .555 / .834 | .863 / .752 / .431 / .737 | **.906 / .865 / .519 / .834** |
| CLINC-150 / HWU64 | .784 / .818 | .740 / .660 | **.862** / .801 |
| MMLU-Pro among-K / acc / ECE | .123 / .092 / .385 | .314 / .133 / .459 | **.363** / .133 / .592 |
| TruthfulQA / kb val | – | .461 / .580 | .348 / .593 |
| CLINC-heldout / 20NG / TREC-fine / Banking77-77 | .762 / .336 / .308 / .527 | .916 / .559 / .400 / .497 | **.955 / .589 / .430** / .063 |
| CLINC-OOS / null AUROC clinc-k / snli | .735 / .961 / .980 | .204 / .803 / .964 | .639 / .840 / .979 |
| P(∅ \| absent) K = 2 / 50 / 150 | .94 / .59 / .45 | .86 / .40 / .99 | .92 / .61 / 1.00 |
| reorder Δp / IIA | 0 / 0 | .101 / .126 | .077 / .136 |
| best val NLL | .371 | .430 | **.341** |

**The evidence gap was depth, not the options-in-suffix formulation.** At tap 20 the native readout is at energy level on
every evidence task (within 1–3 pts; +8 on CLINC-150) *and* keeps all of the question-dependent knowledge: Δ_q(shuffled) **.118** vs .117 for N3 @ 28 (1.16× the teacher;
choices-only variant .100 vs .111). The raw among-K gain (.314 → .363) is mostly higher option-prior exploitation
(choices-only .203 → .263), so read it as "same knowledge, evidence recovered", not "more knowledge". The two-regime story for accuracy (§3l, §3o) collapses into one model; what the energy path
still owns is K-flat cost (§3m) and the null. Remaining defects are exactly the rendered-∅-line ones of §3p (abstention error
.23 on MMLU, Banking77-77 .063, K = 150 → always abstain), which `letters_nonull` addressed in `nc_v3` — `nc_v3_tap20` is
queued as the candidate for a single coherent PCDM v2 model. TruthfulQA fell (.461 → .348) — to be read with the seed pair.
This also reopens §3g ("depth is not the lever"): it was not the lever for the *candidate-blind* readout; for the
candidate-aware readout, depth is the difference between losing and keeping evidence reasoning.

## 3s. Closure — candidate-blind Z at full depth (`e3b_zr_tap28`, H100, 2026-09-20, ~$9)

PLAN6 asked for this as a *closure* experiment, not an improvement attempt: every candidate-blind student in §3j was tapped
at layer 20, and §3r showed depth is decisive for the candidate-*aware* readout, so "Δ_q ≈ 0 for Z" could still have been a
tap-20 artifact. Same `zr` head, same v5 + kbt mix, same KD (α = 1, β = 1, T = 2), same 12k steps, all 28 layers:

| | `e3b_zr` @ 20 | **`e3b_zr_tap28`** | teacher |
|---|---|---|---|
| MMLU-Pro among-K / choices-only / shuffled-q | .157 / .166 / .182 | .147 / .177 / .156 | .310 / .222 / .208 |
| Δ_q (choices-only) / Δ_q (shuffled, primary) | −.009 / −.026 | **−.030 / −.008** | .088 / .102 |
| SNLI / MNLI / ANLI / BoolQ | .905 / .873 / .541 / .826 | .881 / .807 / .462 / .712 | – |
| CLINC-150 / HWU64 / CLINC-heldout | .724 / .784 / .391 | .702 / .787 / .290 | – |
| kbt val (agreement with teacher) | .326 | .314 | – |
| best val NLL | .384 | .463 | – |

**Closed: the candidate-blind negative is not a depth artifact.** With the full backbone the decision state still carries
no question-dependent knowledge (Δ_q_sh −.008, within the ±.02 noise of §3j; among-K *falls* to .147), while the
evidence tasks regress exactly as §3l/§3r predict for a last-layer tap (BoolQ −11, ANLI −8, MNLI −7). Depth therefore
separates the two readouts cleanly: it recovers evidence for the option-conditioned suffix (§3r) and does nothing for a state
computed before the options are known. This is the strongest form of contribution #2 in NOVELTY.md — under this
factorization, priors and calibration distill, question-conditioned parametric knowledge does not, at any tap. No further
candidate-blind runs are planned.

## 3t. One model — native N3 @ tap 20 without the rendered ∅ line (`nc_v3_tap20`, H100, 2026-09-20, ~$9)

§3r recovered evidence at tap 20 but kept the rendered "none of the above" pathologies; §3p's `letters_nonull` removed the
∅ line at 28 layers and lost evidence. This run combines them: N3, factored null, `letters_nonull`, tap 20, same v5 + kb
mix and steps. Matched control = `nc_n3_tap20` (identical except `--nc_render letters`).

| | `nc_n3_tap20` (∅ line) | **`nc_v3_tap20`** (no ∅ line) | energy `joint_emb_lw_v5` |
|---|---|---|---|
| SNLI / MNLI / ANLI / BoolQ | .906 / .865 / .519 / .834 | .904 / .865 / .507 / .834 | .909 / .880 / .555 / .834 |
| CLINC-150 / HWU64 / 20NG / TREC-fine | .862 / .801 / .589 / .430 | .845 / .757 / .515 / .468 | .784 / .818 / .336 / .308 |
| Δ_q shuffled (primary) / choices-only | .118 / .100 | **.122** / .098 | ≈0 |
| MMLU-Pro among-K / **acc** / false-abstain | .363 / .133 / .692 | .354 / **.353** / **.003** | .123 / .092 / – |
| MMLU-cf KL to teacher (orig) | 2.43 | **.27** | – |
| Banking77-77 acc / false-abstain | .063 / .924 | **.579** / .135 | .527 / – |
| CLINC-OOS acc / CLINC-K null AUROC | .639 / .840 | **.798** / **.973** | .735 / .961 |
| K-sweep CLINC null AUROC K = 5 / 20 / 50 / 150 | .97 / .94 / .91 / 1.00† | .98 / .95 / .92 / .95 | .98 / .94 / .90 / .85 |
| K-sweep Banking77 null AUROC K = 5 / 20 / 77 | .91 / .80 / .00† | **.90 / .83 / .69** | .85 / .75 / .66 |
| P(∅ \| absent) K = 5 / 50 / 150 (CLINC) | .02 / .61 / 1.00 | .89 / .67 / .64 | .88 / .56 / .41 |
| cse_* paired null AUROC (banking / clinc / hwu64) | .968 / .995 / .987 | .862 / .976 / .926 | – |
| reorder max Δp / IIA dlo | .077 / .136 | .123 / .180 | 0 / 0 |
| TruthfulQA MC1 | .230 | .275 | – |
| best val NLL | .341 | .347 | .371 |

† degenerate: always-abstain (K = 150 CLINC, K = 77 Banking77) or never-abstain (K = 5), see §3p/§3r.

**The rendered ∅ line was the null pathology, and removing it at tap 20 costs nothing on knowledge.** With ∅ a pure head
decision the K-sweep is monotone and smooth (no K = 5 collapse, no K = 77/150 always-abstain), null AUROC now matches or
beats the energy path at *every* K on both sweeps, MMLU-Pro false-abstain drops from .692 to .003 (acc .133 → .353, KL to
the teacher 2.43 → .27), and Banking77-77 goes from unusable to .579 — all with Δ_q unchanged (.122) and NLI/BoolQ at
energy level. The costs are real but smaller: intent/topic label spaces lose 2–7 pts (HWU64 .801 → .757, 20NG .589 → .515)
and the paired choice-set null probes lose 2–10 pts AUROC; order fragility rises (reorder .077 → .123 — still a letter
interface). This is the single-model candidate PLAN5/PLAN6 asked for, and it is the base and the matched baseline for
Phase 6A (§3v).

**JevBench, same public 231 ids (§3q protocol and disclosures):** standard **.694** / easy **1.00** / hard .378
(Brier .47 / .02 / .74; ECE .135 / .057 / .209), versus .472 / .812 / .297 for `nc_v3` at 28 layers and .417 / .854 / .306
for `nc_n3`. The standard gain is 3.8 SE (n = 72, SE .058) and concentrated where the null/letter pathology had been
eating answers: routing 2/12 → 10/12, extraction 6/12 → 12/12, policy 5/12 → 8/12 (ordinal 7 → 5, adequacy 6 → 6).
With **no workflow training** this puts PCDM at the level of the small encoders trained on the task family (Laya .694, GLiNER2 .639)
and above open-jev-deberta (.431); hard is still within 1 SE of chance (.378 vs .336; long_policy .21 → .42, multi_hop 0 → .33,
temporal/probability/tradeoff flat or down). So part of the §3q "training-distribution" gap was our own null artifact;
the remaining standard gap to the workflow-trained systems (.83–.99) and all of the hard gap are what Phase 6A tests.

## 3u. Learned per-input expert gate — negative (`scripts/gate_experts.py`, eval-time on dumped logits, ~$1)

PLAN6 item 3: a tiny gate (193 params, MLP over per-input signals available at inference — energy max-p / entropy /
top-2 margin / P(∅), the same four for native, log K, log query length; `energy_only` and `native_only` ablations
with 129 params) mixes the energy expert `joint_emb_lw_v5` and the native expert `nc_n3` in log-space, trained on the
v5 val logits only (no task ids), evaluated on 31 held-out sets. The number that decides whether routing is worth
anything is the oracle envelope P(either expert correct) against the best single expert *per set* (which itself
needs a task id) and the gate.

| mean accuracy over the 31 sets | energy | native `nc_n3` | best single expert per set | **learned gate** | oracle (either correct) |
|---|---|---|---|---|---|
| all features | .736 | .700 | .769 | **.745** | .861 |
| energy-only / native-only features | | | | .742 / .743 | |

Val NLL .368 (g = 0) → .345 (gate) → .338 (+T). The gate recovers **+0.9 of the +12.5 envelope** and stays 2.4 pts below
the per-set best expert: where native is clearly better (CLINC-heldout .916 vs .762, 20NG .559 vs .336, TREC-coarse
.630 vs .356) the gate sits near the energy number (.772 / .455 / .422) — its mean g on those sets is .13–.52, i.e. the
per-input signals do not identify "this is a label-space decision the native reader should own". The three feature sets
are indistinguishable (±.003). The envelope is large (+12.5), so complementarity is real, but it is not recoverable from
confidence statistics; a task id would be needed, and that is not a System-One primitive. Not promoted — and with
§3t's `nc_v3_tap20` at energy level on evidence, the two-expert frontier this gate was meant to exploit has mostly
closed on its own (the remaining energy-only advantages are K-flat cost and the paired-null probes).

## 3v. Seed pair for the N1-vs-N3 claim (`nc_n1_s1`, `nc_n3_s1`; H100, 2026-09-20, ~$18)

§3l rested on one seed per readout. Matched seed-1 replicates (28 layers, v5 + kb, 12k steps; only `--seed 1` changed):

| | N1 letters s0 / **s1** | N3 direct s0 / **s1** | teacher |
|---|---|---|---|
| Δ_q shuffled (primary) | .114 / **.104** | .117 / **.094** | .102 |
| Δ_q choices-only | .112 / .093 | .111 / .070 | .088 |
| MMLU-Pro among-K | .325 / .318 | .314 / .301 | .310 |
| SNLI / MNLI / ANLI / BoolQ | .838 / .684 / .377 / .769 → .831 / .664 / .393 / .764 | .863 / .752 / .431 / .737 → .851 / .719 / .437 / .746 | – |
| CLINC-150 / HWU64 | .697 / .510 → .708 / .532 | .740 / .660 → .741 / .674 | – |
| reorder max Δp / IIA dlo | .165 / .463 → .165 / .443 | .101 / .126 → .109 / .094 | 0 / 0 |
| MMLU false-abstain | .254 → .297 | .646 → .543 | – |

Seed-to-seed spread of Δ_q_sh is ≈ .01–.02 for both readouts; N1 and N3 stay within it of each other and of the
teacher (N1 mean .109, N3 mean .106, teacher .102). The evidence advantage of N3 over N1 (SNLI +2, MNLI +6–7, ANLI +4–5,
CLINC +3–4, HWU64 +14–15) and its lower order fragility (reorder .10–.11 vs .165; IIA .09–.13 vs .44–.46) hold in both
seeds. The §3l claim — *the direct contextual readout keeps the letter interface's question-dependent knowledge while
removing most of its order artifacts* — is replicated. MMLU false-abstain moves ±.05–.10 between seeds for both
readouts, so §3p/§3t's abstention numbers should be read at that resolution (the .69 → .003 change of §3t is far outside it).

## 5. Phase-4 log (all items below are complete as of 2026-09-18; kept as the chronological record — current status is in PROJECT.md)

- `joint_v1` — **done** (§3b). Decision rule (SNLI ≥ 80) met with margin.
- `tower_big` — R2 control, **done**: 4×1024-d tower with separate encoding is *worse* than 2×512 (SNLI 64.1 vs 68.5, val NLL
  0.83 vs 0.70). Capacity was never the bottleneck; the pre-registered read-out (R1 ≥ 80 and R2 ≤ 72) holds on both legs.
- Closure checks (2026-09-18): real-model KV-cache `decide` == concatenation path (Δ ≤ 1e-3 over chunk sizes 1/2/3/5/64;
  permutation-consistent; repeat-exact) → no per-query state recomputation, no query↔query leakage. Near-duplicate leak
  audit (MinHash, char-5-gram) running; Codex demanded two stronger "Qwen does everything" baselines (native candidate
  log-probs with the same joint prefix + val-fitted null threshold, uncapped; candidate-blind per-family heads on the
  joint representation) — queued as eval-only runs on the joint_v1 checkpoint.
- Queued (IDEA2 Phase 1): `bench_joint` (H2 with KV-cache queries), `joint_v1_s1` (second seed), `joint_24k` (2 epochs),
  `abl_notower` (direct head vs tower). Cost guard (`costguard.sh`) stops idle pods every 10 min.
- **`abl_notower` (done, 2026-09-18): the cross-attention slot is removed from the design.** Same data/steps as `joint_v1`,
  tower replaced by mean-pooled (state-conditioned) query tokens → same scorer/null. NLI unchanged (90.8/87.7/54.7 vs
  91.0/87.4/54.4); CLINC K=150 **78.4 vs 68.8**, HWU64 86.2 vs 81.6, unseen-intent among-K 0.80 vs 0.74, CLINC-OOS null recall
  **65.7% vs 32.3%**, null AUROC .946 vs .902; best val NLL of any run (0.389 vs 0.419). The joint backbone does the
  interaction; the slot only added a harder-to-train bottleneck. Default is now `--tower_layers 0`.
- **`joint_v2` (data v4, no tower) — done.** In-distribution unchanged; null cleaner (CLINC-OOS recall 72.3%, ECE 0.007;
  new slices: irrelevant-question AUROC .99/.91, near-miss .91/.96). Choice-set battery: independent scoring is exactly
  IIA (Δlog-odds = 0 under add-irrelevant), reorder exact, duplicate mass error 2–4%. **But P(∅|gold absent) still falls
  0.98 → 0.35 from K=2 to K=150 (range 0.63)** with nulls present at every K in training — the K-dependence is structural
  (softmax dilution), not a data prior. That is the test for the listwise head (`joint_lw`, target range ≤ 0.10).
- Review (REVIEW.md) done; Phase 2 in flight: `joint_v1_s1` (seed), `mcq_lora` (the "Qwen does everything" baseline),
  `zs_mcq_8B`, `bench_fair`, `joint_v2` (data v4 control, no tower), `joint_lw` (listwise), `joint_emb` (embedding-model
  candidates — the $0 probe showed the candidate encoder is the unseen-label bottleneck).

## 6. What's next (not done)

Label-space-held-out training splits (the §3d fix for false null on unseen vocabularies). Cross-encoder distillation into the joint model (+1–3, literature), a second joint seed, 8B prompted baseline with a proper
null protocol instead of a literal string, the `Score` type, uncertainty-shaping losses (§12) and the workflow-level H5.

## 7. Decision log (why things were done)

- 2026-09-16 v0 on Mac: frozen 0.6B + cached features so every ablation was minutes; found the sink-token, fixed-classifier
  and null-leak bugs; H2 shape confirmed. Decided a GPU phase was needed because both F and C sat at the frozen-feature ceiling.
- 2026-09-17 v1 design (deep-reasoner + Codex): LoRA top-8 on 1.7B, label-diverse data, hybrid scorer, H100 ≤ $50.
  Local gate (16 tests, smoke, mini run) before renting; GPU smoke caught CPU-bound collate + fp32 LoRA copies.
- `main_s0` failed H1 (58.6). Diagnosis chain: hypothesis-only probe → last-layer tap → tap 20; rogue dims → z-score;
  null over-firing → data v3. Cut `main_4B` (scale was not the bottleneck) and ablations the diagnostics had answered.
- Second pod added to halve wall-clock (same cost); cost $ ≈ $37 by end of phase 3.
- Literature review (Poly-encoders, ColBERT, DeFormer, PreTTR, LUMEN) + Codex review converged: the cross-encoder is itself
  late-interaction (causal); interaction must run through pretrained layers → `--joint` (bit-exact causal invariance test).
- Rejected for now: distillation (second-order once joint works), deeper tower (control only), two-layer taps, KDA (IDEA2 §18).
- Open questions carried into the review: is the tower needed after joint encoding; listwise candidates for the choice-set
  effects and unseen-label sets; ChaosNLI sharpness vs calibration (Stage B); TREC/20NG remain < 20% for every model.
