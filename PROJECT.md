# PCDM — project entry point (2026-09-18)

Read this first. It indexes every document, file, run and finding of the project so far; it does not repeat their
contents. Numbers quoted here are copied from `REPORT.md` (the consolidated results doc) and `runs/*/results.json`.

## 1. What PCDM is, and where it stands

PCDM (Parallel Calibrated Decision Model) turns a pretrained LM into a direct probabilistic decision engine
`F(x, q, A) → P(y | x, q, A)`, `y ∈ A ∪ {∅}`: an unstructured state `x` is encoded once, then many natural-language
queries `q_i` with runtime-defined candidate sets `A_i` are answered in parallel as calibrated distributions with an
explicit "none of the above" (∅). No generation, no JSON parsing, no per-query re-encoding of the state.

**Headline (REPORT §3b–§3d, H1–H5 scorecard §4).** The current best model is the *joint* family: Qwen3-1.7B-Base
truncated at layer 20, LoRA r16 on layers 13–20, query encoded as a causal suffix over the cached state (KV cache at
inference), no cross-attention tower, candidates from Qwen3-Embedding-0.6B, energy scorer + candidate-aware null.

| | `joint_emb` (best large-K acc) | `joint_emb_lw` (best null) | fine-tuned cross-encoder `C_lora` | prompted 8B `B_8B` |
|---|---|---|---|---|
| SNLI / MNLI / ANLI / BoolQ | .906 / .873 / .524 / .817 | .905 / .873 / .521 / .815 | .855 / .768 / .376 / .598 | .818 / .823 / .504 / .863 |
| CLINC-150 / HWU64 | .803 / .875 | .763 / .869 | .393 / .623 | .387 / .020 |
| paraphrased labels / Banking77-77 (unseen) | .810 / .440 | .820 / .436 | (.841) / – | .016 / .200 |
| null AUROC clinc-k / snli / squad | .953 / .979 / .976 | .960 / .980 / .978 | – | .617 / .887 / .886 |
| CLINC-OOS null recall | .649 | **.823** | – | .960 (literal string) |
| SNLI ECE / fitted T | .005 / 0.96 | .006 / 1.04 | .036 / 0.96 | .494 / 4.58 |
| per-query cost vs prompted LM, M=256 (`bench_fair`) | 73× / 37× / 20× / 16× cheaper at K = 4 / 32 / 150 / 1000 | | | |

Scorecard: **H1** holds for the joint family (fails for the v1 tower family); **H2** holds (fair bench); **H3** holds
in-distribution (OOD ECE .19–.69, REVIEW §2); **H4** null holds, unseen label spaces partial — the residual gap is a
*novelty ⇒ null* confound, not a K-bias (REPORT §3d); **H5** (listwise) partial — halves K-dilution of the null
(range .67 → .43, target ≤ .10), costs −4/−7 on CLINC-150, breaks IIA by design. Seed floor: NLI ±0.7, trained
large-K ±3, unseen label spaces / OOS ±5 (REPORT §3d(3)).

In progress (do not edit, link only): `COMPARE.md` (PCDM vs TypeSafe JEV, agent `a-compare`) and data v5 in
`data.py` (label-space-held-out training splits, worker `w-datav5`); a K-aware null-bias follow-up (`w-nullbias`).

**2026-09-19 redirect:** see `PLAN4.md` (closed-set native choice, N1/N2/N3 readouts) and `REPORT.md §3j–§3k`; E3/E3-ms becomes the optional candidate-blind compilation branch. `report_native.py` = PLAN4 §15 key table. **Outcome (2026-09-20): native choice adopted — `nc_n3` matches the letter readout on knowledge (MMLU-Pro .314, Δ_q .111) without generation; see PLAN4 §21 and REPORT §3l. All pods deleted; budget ≈ $5 left.**

## 2. Document map (read in this order)

| doc | what it is | read when |
|---|---|---|
| `idea.md` | v0 statement of the thesis: §1 core equation, §2 desired properties, §3–§16 architecture/training stages, §17 H1–H5, §18 baselines, §20 v0 experiment | you want the original motivation and the pre-registered hypotheses |
| `IDEA2.md` | PI's revised statement after v1: §3 "pretrained state–query interaction is load-bearing" (joint prefix architecture), §6 listwise candidates, §8–§12 training stack (Stage A/B/C, RLCD), §14–§19 KDA (deferred), §20 H1–H7, §21 roadmap Phases 1–5 | you want the current architectural position and the full hypothesis stack |
| `PLAN.md` | v0 PoC plan (Mac/MPS, frozen 0.6B): files, JSONL schema, data mix, interfaces, decision rules for H1–H4 | you need the data schema or the v0 rules |
| `PLAN2.md` | v1 plan (RunPod H100): LoRA/backbone decisions, data v2 spec, run schedule, pre-registered rules, local gate, RunPod mechanics, **"Revision after main_s0"** (tap layer, z-score, data v3) | you are about to rent a GPU or change the training config |
| `REVIEW.md` | stop-and-rethink review after `joint_v1`: §1 what stands, §2 what does not (baseline fairness, K–null coupling, OOD calibration), §3 the "fancy classifier over Qwen" question → MCQ-LoRA, §4 cuts, §5 ranked plan + §5a cosine probe + §5b outcomes, §6 threats to validity | you want to know *why* Phase 2 ran what it ran and what is still unproven |
| `REPORT.md` | consolidated results: §1 what was built, §2 bugs/findings in order, §3 v1 tower results + H2, §4 scorecard, §3b joint_v1, §3c Phase 2 (MCQ-LoRA, joint_v2/lw/emb, fair bench), §3d joint_emb follow-ups (null bias, emb+lw, seed), §5 pending, §6 next, §7 decision log | you need a number, a verdict, or the reason a decision was taken |
| `RESULTS.md` | v0 snapshot table (frozen 0.6B runs B/C/Clate/D/E/F/G, 3 seeds) + v0 H2 bench + v0 verdicts | historical only; superseded by REPORT §3 onwards |
| `README.md` | v0 README: v0 run commands, v0 findings, v0 verdicts | historical only; the CLI it shows is stale (see Doc debt) |
| `COMPARE.md` | **in progress** — PCDM vs TypeSafe JEV comparison | when it lands |
| `PROJECT.md` | this file | first |

## 3. Code map

All Python runs through `uv` (`uv run ...`; on pods `uv run --no-sync ...`). Tests: `tests/test_pipeline.py` (29 tests,
CPU, tiny random-weight Qwen3 from `conftest.py`).

| file | role | key entry points / flags | covered by |
|---|---|---|---|
| `encode.py` | backbone + LoRA + caches. `Backbone(name, lora_layers, lora_r, tap_layer)` (hand-rolled `LoRALinear`, `tap_layer` truncates the model, `frozen_features` from `split_layer = tap − lora_layers`), `FeatureCache` (token features per string), `EmbedEncoder`/`VecCache` (Qwen3-Embedding-0.6B, `EMBED_DIM=1024`), `build_cache`, `build_vec_cache`, `pick_device` | `uv run encode.py` (v0 cache build); `python -c 'import encode; encode.selftest()'` | `test_lora_zero_init_and_grads`, `test_joint_causal_invariance`, `test_embed_encoder_unit_norm` |
| `model.py` | `DecisionModel` (optional cross-attn tower `num_layers`, hybrid frozen-space sim features, candidate-aware null, optional `SetMixer` listwise block, `d_cand` for embedder candidates), `decision_loss` (soft-CE over `[(1−p∅)·target, p∅]`), `collate`, `run_batch(joint=…)`, `split_joint`, `decide` (encode state once, chunk M queries, KV cache) | library only | `test_collate_ragged`, `test_permutation_equivariance`, `test_loss_matches_hand_soft_ce`, `test_null_target_all_mass_on_null`, `test_run_batch_end_to_end`, `test_decide_matches_batched`, `test_joint_run_batch_shapes`, `test_decide_joint_matches_run_batch`, `test_candidate_permutation_equivariance`, `test_padding_invariance`, `test_duplicate_slots_identical`, `test_listwise_identity_at_init_and_warm_start`, `test_add_candidate_odds`, `test_decision_model_d_cand_forward_backward`, `test_run_batch_with_fake_vec_cache` |
| `train.py` | train + eval + temperature fit + `results.json` + W&B + HF upload; auto-resume from `last.pt`; `--eval_only` rebuilds the architecture from `best.pt`'s saved args | `--joint --tower_layers 0 --tap_layer 20 --zscore --lora_r 16 --cand_encoder qwen3emb --listwise --readout energy\|mcq --zero_shot --init_from --eval_only --dump_logits DIR --smoke --data DIR --steps 12000 --bs 64 --seed` | `test_checkpoint_resume_reproduces_loss`, `test_wandb_offline`, `test_null_bias_real_path_on_mini_v4` |
| `data.py` | HF datasets → JSONL (`train/val/eval/*.jsonl`, `held_out_intents.json`); v4 scheme = K-decoupled nulls + `cse_*`/`ksweep_*`/`null_irrq_*`/`null_nearmiss_*` eval sets; `TRAIN_NAMES`/`PARAPHRASES`; `selftest()` leak audit; **v5 being added by `w-datav5`** | `uv run data.py [--small] [--out DIR]`, `uv run data.py --selftest --out data_v4` | `test_data_audit_if_present` |
| `mcq.py` | `MCQHead` = same backbone/LoRA, answer read from next-token letter logits over enumerated options + "none" letter; chunked two-stage above 51 options | via `train.py --readout mcq` | `test_mcq_letter_logits_shape_finite`, `test_mcq_padded_columns_are_finfo_min`, `test_mcq_chunked_k70` |
| `metrics.py` | `summarize` (acc, acc_k, nll, brier, jsd, ece, auroc_null, auroc_conf, sel_acc80, conf_wrong), `choice_set_effects` (cse battery), `ksweep` (P(∅\|absent) vs K, range) | library | `test_metrics_summarize_hand_values` |
| `baselines.py` | B = prompted log-prob with KV-cached prefix (`--backbone`, `--limit`, `--check`); C = LoRA cross-encoder + per-task heads | `uv run baselines.py B --backbone Qwen/Qwen3-8B-Base --data data_v3 --name B_8B`; `uv run baselines.py C --name C_lora` | none (`--check` self-check only) |
| `bench.py` | H2 microbench: ours vs B vs B_fair (prefix shared), K ∈ {4,32,150[,1000]} × M ∈ {1,8,64,256}; writes `bench.json` + `results.json` | `uv run bench.py --model runs/joint_v1 --name bench_fair [--full] [--quick]` | none |
| `report.py` | side-by-side table of `runs/*/results.json` on the hypothesis columns | `uv run report.py [--raw]` | `test_report_handles_na` |
| `scripts/cosine_probe.py` | $0 probe: untrained nearest-label cosine on held-out label spaces for 3 candidate encoders (REVIEW §5a) | `uv run scripts/cosine_probe.py [--limit N]`, `--selftest` | — |
| `scripts/leak_audit.py` | MinHash char-5-gram near-dup audit train vs every eval set → `runs/leak_audit.json` (+ per-set near-dup lists) | `uv run scripts/leak_audit.py` | — |
| `scripts/null_bias.py` | post-hoc `s∅' = s∅ + α·log K + β` fitted with T on val from a `--dump_logits` dir (REPORT §3d(1)) | `uv run scripts/null_bias.py runs/dump_joint_emb --out runs/joint_emb_nullbias/results.json` | `test_fit_null_bias_finite_and_improves_val_nll`, `test_fit_null_bias_three_ragged_sets_end_to_end` |
| `conftest.py` | tiny Qwen3 backbone / cache fixtures | — | — |
| `costguard.sh` | every 10 min: stop a RunPod pod idle (no train/baselines/bench process, GPU < 5 %, no waiter) for 2 checks; pod ids hard-coded in `PODS` | `./costguard.sh &` (log `/tmp/costguard.log`) | — |
| `watchdog.sh` | v0 Mac helper: rerun a command if its log stops growing for 5 min (MPS stream hang) | `./watchdog.sh LOG CMD...` | — |
| `run_gpu.sh` / `run_gpu2.sh` / `run_gpu3.sh` / `run_gpu3b.sh` | idempotent pod schedules for phases 1 / diag / 3 (pod 1) / 3 (pod 2); skip a run if `runs/<name>/results.json` exists; `timeout Nh` + `runpodctl pod stop` | `tmux new -d 'bash run_gpu3.sh'` on the pod | — |
| `run_rest.sh` | v0 Mac schedule (B, cache, F/G/D/E eval, bench, seeds) | historical | — |
| `pod_setup.sh` | fresh-pod bootstrap: env dirs on `/workspace`, uv, tmux, `hf download` data, 300-step GPU smoke | `bash pod_setup.sh` | — |

Phase-2 schedules (joint_v1, tower_big, joint_emb_lw, joint_emb_s1, dump) were one-off scripts sent to the pods
(`/tmp/pod{1,2,3}_*.sh` on the Mac, not in the repo); their exact commands are reproduced below.

### Exact commands

```bash
# tests (CPU, < 60 s)
uv run pytest tests/ -q

# data: current data.py builds the v4 scheme (33 eval sets). v3 = 22 sets (data/), v4 = data_v4/
uv run data.py --out data_v4 && uv run data.py --selftest --out data_v4
uv run data.py --small                          # 200/source -> data_small/ (local gate)
uv run hf download guychuk/pcdm-data --repo-type dataset --local-dir data_v3          # v3 at repo root
uv run hf download guychuk/pcdm-data --repo-type dataset --include 'v4/*' --local-dir .   # v4 under v4/
uv run scripts/leak_audit.py                    # -> runs/leak_audit.json

# train the current default config (= runs/joint_emb_lw; drop --listwise for joint_emb, add --seed 1 for joint_emb_s1)
uv run python train.py --name joint_emb_lw --joint --tower_layers 0 --tap_layer 20 --zscore --lora_r 16 \
    --data data_v4 --cand_encoder qwen3emb --listwise --eval_every 6000 --wandb --hf_repo guychuk/pcdm-runs

# eval-only from best.pt (architecture flags are read from the checkpoint) + logits dump
uv run python train.py --name joint_emb --eval_only --data data_v4 --dump_logits runs/dump_joint_emb

# fit the K-aware null bias on the dump (no retraining) -> results.json in train.py's schema
uv run scripts/null_bias.py runs/dump_joint_emb --out runs/joint_emb_nullbias/results.json

# side-by-side table
uv run report.py            # scaled (T fitted on val);  --raw for unscaled

# fair H2 bench (ours vs B vs B_fair with the state prefix shared; --full adds K=1000)
uv run bench.py --model runs/joint_v1 --backbone Qwen/Qwen3-1.7B-Base --name bench_fair --full

# baselines
uv run baselines.py B --backbone Qwen/Qwen3-1.7B-Base --data data_v3 --name B_1.7B
uv run baselines.py B --backbone Qwen/Qwen3-8B-Base   --data data_v3 --name B_8B
uv run baselines.py C --name C_lora
uv run python train.py --name zs_mcq_8B --zero_shot --readout mcq --backbone Qwen/Qwen3-8B-Base --lora_r 0 --data data_v3
uv run scripts/cosine_probe.py
```

Smoke / mini gates before renting: `uv run train.py --name smoke --smoke --wandb`, then
`uv run train.py --name mini_v4 --steps 300 --bs 16 --backbone Qwen/Qwen3-0.6B-Base --data data_small_v4`.

## 4. Run registry

Every `runs/<name>/results.json` (local); `best.pt` for GPU runs lives only on HF `guychuk/pcdm-runs` (private) and
the pod volume. W&B: `guy-na8/pcdm`. "REPORT" = section of `REPORT.md`. Data: v1 = v0 Mac mix (44k), v2 = PLAN2 mix
(758k), v3 = v2 with fixed null synthesis (PLAN2 revision), v4 = K-decoupled nulls + cse/ksweep/null-slice eval sets.

### v0 — Mac, frozen Qwen3-0.6B-Base layer-20 features, 9M tower (RESULTS.md, README.md, REPORT §2.1–2.3, §7)

| run | config | data | seed | question | outcome |
|---|---|---|---|---|---|
| `F` (+`F_s1`,`F_s2`) | tower, soft labels, state-only null | v1 | 0/1/2 | v0 "the model" | SNLI .667 ±.001, ChaosNLI 1.35: state-only null leaks its base rate (H3 fails) |
| `G` (+`G_s1`,`G_s2`) | F + candidate-aware null | v1 | 0/1/2 | fix the leak | NLL 1.15, ECE .034 → cand-null becomes default |
| `D` | no null | v1 | 0 | ablation | NLL 1.17; no null column |
| `E` | hard labels only | v1 | 0 | ablation | soft labels alone buy nothing |
| `C` | frozen cross-encoder + linear heads | v1 | – | accuracy ceiling | SNLI .684 (frozen-feature ceiling) |
| `Clate` | pooled MLP, no cross-attn | v1 | – | is the tower needed | −8 NLI but better unseen-intent transfer (frozen-space match) |
| `B` | prompted 0.6B log-probs | v1 | – | baseline | SNLI .45; 0 % on paraphrased labels |

### v1 tower family — RunPod, Qwen3-1.7B-Base + LoRA, separate state/query encoding (REPORT §2.4–2.7, §3, §4; PLAN2 revision)

| run | config | data | seed | question | outcome |
|---|---|---|---|---|---|
| `main_s0` | last-layer tap, LoRA top 8 | v2 | 0 | H1 at scale | SNLI .586 = hypothesis-only → tap was wrong (REPORT §2.4) |
| `diag_tap20_frozen` | tap 20, `--lora_r 0` | v2 | 0 | is it the tap | .653 |
| `diag_tap20` | tap 20 + LoRA 13–20 | v2 | 0 | is LoRA needed | .682 → both adopted |
| `main_v2` | tap 20 + LoRA + `--zscore` | v2 | 0 | rogue dims | .681, +2–4 CLINC/HWU, best null AUROC (REPORT §2.5) |
| `main_v3` | main_v2 + data v3, 24k steps | v3 | 0 | null over-fire fix | .701 / CLINC .708; TREC/20NG doubled (REPORT §2.6); the tower reference |
| `main_v3_s1` | same, 12k | v3 | 1 | noise | .685 (not a clean pair: 1 vs 2 epochs) |
| `abl_nohybrid` | main_v3 − frozen-space sim | v3 | 0 | unseen-label mechanism | Banking77 .33 → .08: hybrid sim is the whole transfer mechanism |
| `tower_big` | 4×1024-d tower | v3 | 0 | R2 control: capacity | worse (.641); capacity was never the bottleneck (REPORT §5) |

### joint family — query as causal suffix over the cached state (`--joint`; REPORT §2.8, §3b, §5)

| run | config | data | seed | question | outcome |
|---|---|---|---|---|---|
| `joint_v1` | `--joint --tap_layer 20 --zscore --lora_r 16`, 2-layer tower | v3 | 0 | R1: pretrained interaction | SNLI .910 / MNLI .874 / ANLI .544 — H1 holds, hyp-only probe 44 % |
| `joint_v1_s1` | same | v3 | 1 | clean seed pair | .908; NLI ±0.7, held-out spaces ±3–5 |
| `abl_notower` | joint, `--tower_layers 0` | v3 | 0 | is the slot needed | NLI equal, CLINC +10, OOS recall +33 → tower removed (REPORT §5) |

### phase 2 — after REVIEW.md (REPORT §3c, §3d; REVIEW §5b)

| run | config | data | seed | question | outcome |
|---|---|---|---|---|---|
| `mcq_lora` | `--readout mcq`, same backbone/LoRA | v3 | 0 | "Qwen does everything" | loses in-dist. (−5 SNLI, −15 MNLI/ANLI, −28 HWU64) and null (.68); **wins unseen label spaces** (+28 CLINC-heldout, +27 Banking77) |
| `zs_mcq_8B` | zero-shot 8B, enumerated letters | v3 | – | learned vs literal null | at chance (base model can't use letters); `B_8B` stays the reference |
| `joint_v2` | notower joint on data v4 | v4 | 0 | K–null decoupling | in-dist. unchanged, OOS recall .72, but P(∅\|absent) .98 → .35 over K: dilution is structural |
| `joint_lw` | joint_v2 + `--listwise` | v4 | 0 | H5 | range .63 → .37, OOS .85; −3–7 large-K; IIA broken by design → partial |
| `joint_emb` | joint_v2 + `--cand_encoder qwen3emb` | v4 | 0 | candidate-encoder bottleneck (REVIEW §5a) | best single change: paraphrase .81, Banking77 .44, ECE .005 |
| `joint_emb_lw` | joint_emb + listwise | v4 | 0 | emb + set view | OOS +17/+22, range .67 → .43, CLINC-150 −4/−7; **default for null-critical use** |
| `joint_emb_s1` | joint_emb | v4 | 1 | seed floor, emb family | ≤ .007 NLI, .026 CLINC, .05 CLINC-heldout/OOS |
| `joint_emb_nullbias` | joint_emb + val-fitted (α, β, T) = (0, −0.25, 0.96) | v4 | 0 | K-aware null bias | no-op on val; cross-OOD fits flip sign → novelty ⇒ null confound (REPORT §3d(1)) |
| `dump_joint_emb/` | `--eval_only --dump_logits` of joint_emb | v4 | – | input for null_bias.py | 34 `.npz`+`.meta.json` pairs (gitignored) |

### baselines, bench, diagnostics

| run | what | data | discussed | outcome |
|---|---|---|---|---|
| `B_1.7B`, `B_8B` | prompted log-prob, KV-cached prefix, 3-shot, literal "none of the above" | v3 | REPORT §3, REVIEW §2 | 8B: SNLI .818, HWU64 .02, T ≈ 4.6; fairness caveats in REVIEW §2 |
| `C_lora` | LoRA cross-encoder + per-task heads, 3k×32 | v3 | REPORT §3, §2.8 | SNLI .855; no null, fixed label sets |
| `bench/` (log only) | H2 tower vs B on `main_v3` | – | REPORT §3 H2 | 69× / 35× / 19× at K = 4 / 32 / 150 |
| `bench_fair` | H2 joint_v1 vs B vs B_fair | – | REPORT §3c | 73× / 37× / 20× / 16× at K = 4 / 32 / 150 / 1000 |
| `leak_audit.json` (+2 nearDup files) | MinHash near-dup train↔eval | v3 | REVIEW §6 | NLI pair-level 1/3/0 rows; BoolQ 4.9 %, HWU64 5.4 % state near-dups (pruned in v4) |
| `mini_v4`, `mini_emb`, `smoke_emb`, `smoke_mcq`, `zs_mcq_small` | local gate runs (0.6B, data_small) | small | PLAN2 local gate | pipeline checks only; numbers meaningless |

## 5. Findings ledger (what changed a decision)

1. **Attention-sink token** — Qwen3 position 0 is content-independent; single-token candidates collapsed. Prepend `<|endoftext|>`, drop it. REPORT §2.1, README.
2. **Fixed classifier in disguise** — three fixed NLI strings → 11 % on paraphrased labels. Randomised label wordings in training, 5 wordings held out. REPORT §2.2.
3. **State-only null leaks its base rate** — 13 % on every clean item. Candidate-aware null (G). REPORT §2.3, RESULTS verdicts.
4. **Last-layer tap = hypothesis-only model** — SNLI .586 at scale; tower never read the premise. `--tap_layer 20`, LoRA below the tap, layers 21–28 dropped. REPORT §2.4, PLAN2 revision 1.
5. **Rogue dimensions** — 3 dims ≈ 30 % of token norm. `--zscore`. REPORT §2.5, PLAN2 revision 2.
6. **Null over-fires from K = N−1 near-miss synthesis** — data v3 (15 %, K ∈ {3,10}). REPORT §2.6, PLAN2 revision 3.
7. **LoRA dropout active at eval** — `backbone.eval()` in eval loops. REPORT §2.7.
8. **The cross-encoder is itself late-interaction** — interaction must run through pretrained layers → `--joint` (bit-exact causal-invariance test). REPORT §2.8, §3b; IDEA2 §3.
9. **Capacity was never the bottleneck** — `tower_big` worse than 2×512. REPORT §5.
10. **Tower removal** — after joint encoding the cross-attn slot only adds a bottleneck; `--tower_layers 0`. REPORT §5 (`abl_notower`).
11. **K–null coupling in training data** — K ≥ 50 rows never null, NLI K=2 50 % null. Data v4 decouples; but dilution remains structural (softmax). REVIEW §2, §5 item 6; REPORT §5 (`joint_v2`).
12. **Candidate encoder is the unseen-label bottleneck** — untrained Qwen3-Embedding cosine beats the trained model on every held-out space; `--cand_encoder qwen3emb`. REVIEW §5a; REPORT §3c.
13. **Decision head vs letter readout** — MCQ-LoRA loses in-distribution, null and cost but wins unseen label spaces; the head is not decorative, stop-rule not triggered. REVIEW §3, §5b; REPORT §3c.
14. **Listwise trade** — SetMixer halves K-dilution and lifts OOS recall, costs large-K accuracy and IIA. REPORT §3c, §3d(2).
15. **Novelty ⇒ null confound** — false null on unseen vocabularies is not a K-bias: fitted β flips sign with the fitting set; needs label-space-held-out training rows (→ data v5). REPORT §3d(1); REVIEW §5b.
16. **Seed floor** — NLI ±0.7, trained large-K ±3, unseen spaces / OOS ±5; several §3c "+2–4" readings demoted to ties. REPORT §3d(3).
17. **Baseline fairness** — B scored labels without the query or other options; prefix-sharing barely helps it (cost is K continuations). REVIEW §2; REPORT §3c (`bench_fair`).
18. **Near-duplicate leakage** — NLI headline is not memorisation; BoolQ/HWU64 ≤ 1–2 pts inflated. REVIEW §6.
19. **Ops** — see §6 below. REPORT §2.9.

## 6. Ops

**Infra.** RunPod Secure Cloud H100 SXM (≈ $3/h), 100 GB network volume at `/workspace` (HF_HOME, UV_CACHE_DIR,
data, runs, `.env`), template `runpod/pytorch`. Pods used: pod 1 (phases 1–3, `run_gpu*.sh`), pod 2
(`3jzl2m7ligff7v`, phase 3b / `joint_v1` / `tower_big`, shared volume), pod 3 (`jd2ali5gufk9fh`, phase 2 review runs,
`joint_emb_lw`, `joint_emb_s1`, logits dump; busy at time of writing). SSH info files `/tmp/podssh{,2,3}` on the Mac.
Artifacts: W&B `guy-na8/pcdm`; HF `guychuk/pcdm-data` (v3 at root, `v4/`, `v5/` soon); HF `guychuk/pcdm-runs`
(every run's `best.pt` + `results.json`). Secrets in `.env` (never print).

**Cost.** v0 $0 (Mac). v1 ≈ $35 for 16 runs, ≈ $37 by end of phase 3 (REPORT §1, §7); REVIEW plan ≈ $45; §3d
follow-ups ≈ $10. Total ≈ $68 of the $145 balance; **≈ $77 remaining**.

**Cost guard.** `costguard.sh` polls each pod every 10 min and stops it after 2 idle checks (no trainer process, GPU
util < 5 %, no waiter script). Pod ids and ssh files are hard-coded in `PODS` — update before a new pod. Schedules
also self-stop (`timeout Nh ...; runpodctl pod stop $RUNPOD_POD_ID`).

**Known pitfalls** (REPORT §2.9; PLAN2 "RunPod mechanics"; also in the user's memory notes `remote-job-hygiene.md`,
`mps-sync-deadlock.md`, `decoder-features-for-heads.md`):
- `tmux kill-session` does not kill `timeout`-wrapped schedule children → ghost schedule trains a second model on the
  same GPU. Kill the `timeout`/`bash -c schedule` pids first (see `/tmp/pod1fix.sh` pattern in §3).
- `pkill -f` over SSH matches and kills the SSH shell itself; use `pgrep -f` + explicit `kill`.
- `runpodctl --env` variables do not reach SSH sessions; source `/workspace/.env` in the script.
- Two pods sharing one uv `.venv` on the volume corrupt it (`uv run` auto-syncs) → `uv run --no-sync` + per-pod envs.
- Driver-570 hosts need a cu126 torch wheel.
- Mac/MPS: boolean indexing / per-item `.item()` hangs under GPU contention; `watchdog.sh` restarts a stalled log.
- `--eval_only` takes architecture flags from `best.pt`, not the CLI; `--readout mcq` rejects `--zscore/--joint/--listwise`.
- `runs/dump_*/` and `*.pt` are gitignored; `best.pt` for GPU runs is on HF only.

## 7. Open questions and next steps (by information per dollar; budget ≈ $77)

1. **Data v5 + `joint_emb_v5`** (~$5 build, ~$4 train) — label-space-held-out training splits so the model sees
   gold-present rows over unfamiliar vocabularies; falsifies "false null on unseen spaces is fixable by training signal"
   (REPORT §3d, §6). Success: CLINC-heldout acc → among-K gap < .05 without losing OOS recall. *(w-datav5 in progress.)*
2. **Null-bias follow-up** ($0–1) — whatever `w-nullbias` is testing beyond the falsified scalar fit (REPORT §3d(1)).
3. **Expected-cost / utility evaluation** ($0, REVIEW §5 item 8) — per-row probs exist via `--dump_logits`; asks whether
   calibration differences change decisions at all (H7 precursor, H5 workflow-level in idea.md).
4. **Second seed of `joint_emb_lw`** (~$4) — the listwise null-side effects clear 2× noise on one seed pair; a
   listwise seed pair would make the "default for null-critical use" claim clean.
5. **Label-name overlap audit** ($0) — exact label-string intersection between training intent sets and Banking77
   (REVIEW §6, still open).
6. **Cross-encoder distillation into the joint model** (~$8; REPORT §6) — literature +1–3; second-order.
7. **8B prompted baseline with a proper null protocol** (~$3; REPORT §6) — `zs_mcq_8B` showed the base model can't use
   letters; needs an instruct model or a log-prob null threshold fitted on val.
8. **Stage-B calibration / uncertainty-shaping losses** (IDEA2 §9; REPORT §6) — ChaosNLI sharpness vs calibration,
   OOD ECE .19–.69 (REVIEW §2). Focal / Dirichlet are the cheap options.
9. Deferred by decision: KDA / H6 (IDEA2 §18; REVIEW §4), 4B backbone, `joint_24k`, Stage C workflow RL (H7).

## 8. Glossary

- **state / query / candidates** — `x` (premise, utterance, passage), `q` (natural-language question), `A` (runtime-defined answer strings, K of them).
- **null / ∅** — the architectural "none of the above / insufficient evidence" outcome; last softmax column; `p_null` in the data.
- **cand-null (variant G)** — null logit computed from the state *and* the score-weighted best candidate, vs **state-only null** (F) which leaks the base rate.
- **tower** — the v0/v1 2-layer 512-d cross-attention decoder (query tokens + slot attend to state tokens). Removed in the joint family (`--tower_layers 0`).
- **joint (`--joint`)** — query encoded as a causal suffix of the state inside the pretrained backbone; state KV cache reused across queries; bit-exact with concatenation.
- **late interaction** — state and query encoded separately, interaction only in a small head (v0/v1 tower, Clate, ColBERT-style). The joint model keeps the pretrained layers as the interaction.
- **tap layer (`--tap_layer 20`)** — the backbone layer whose hidden states feed the head; the model is truncated there. Last layer = next-token-shaped = hypothesis-only.
- **split layer** — `tap − lora_layers` (= 12): frozen features for the hybrid similarity and (pre-`qwen3emb`) candidates come from here.
- **LoRA 13–20** — hand-rolled r16 adapters on layers 13–20 (`--lora_layers 8`), layers 1–12 frozen.
- **z-score (`--zscore`)** — per-dimension standardisation of backbone features (stats from 512 train states) to neutralise rogue dimensions.
- **hybrid scorer** — the energy scorer also sees frozen-space `ln(U)⊙ln(C)`, `ln(V)⊙ln(C)`, cosines; the transfer mechanism for unseen labels (`--no_hybrid` ablation).
- **candidate encoder (`--cand_encoder qwen3emb`)** — candidate/state/query vectors for the similarity features from Qwen3-Embedding-0.6B (`VecCache`) instead of mean-pooled decoder features.
- **energy readout vs MCQ readout** — learned per-candidate score + null logit (ours) vs next-token letter logits over enumerated options (`mcq.py`, the "Qwen does everything" baseline).
- **listwise / SetMixer (`--listwise`)** — one zero-init self-attention block over `[h; c_j + sim_j]` so candidates see each other; identity at init; breaks IIA by design.
- **IIA** — independence of irrelevant alternatives: adding an irrelevant option must not change odds between existing ones (Δlog-odds = 0 for independent scoring).
- **cse battery** — choice-set-effects eval sets (`cse_*`): variants add-irrelevant / remove-gold / duplicate / reorder / near-dup of a base row; metrics are mass movements per variant (`metrics.choice_set_effects`).
- **K-sweep (`ksweep_*`)** — same item at K ∈ {2 … 150}, gold present/absent; reports P(∅|absent)@K and `p_null_absent_range` (target ≤ .10).
- **null slices** — `null_irrq_*` (irrelevant question), `null_nearmiss_*` (sibling-label distractors), `clinc_oos`, `snli_null`, `squad_null`.
- **among-K (`acc_k`) vs acc** — accuracy with the null column ignored (right candidate ranked first) vs full accuracy including null; the gap is the false-null rate.
- **false-null gap / novelty ⇒ null confound** — unfamiliar label vocabulary read as "gold absent"; the current bottleneck on unseen label spaces.
- **hypothesis-only probe (`snli_test_hyponly`)** — premise blanked; high accuracy means the model ignores the state.
- **paraphrased labels (`snli_test_paraphrase`)** — 5 held-out label wordings; tests reading the candidate text.
- **held-out label spaces** — Banking77, TREC, 20NG (never trained on) and 30 held-out CLINC intents.
- **T / scaled** — one scalar temperature fitted on val by NLL and applied to every eval set; `results.json` holds raw and scaled.
- **soft-CE** — cross-entropy against `[(1−p∅)·target, p∅]` (ChaosNLI / SNLI-soft / UNLI targets are distributions).
- **data v1…v5** — see §4 header; v5 = label-space-held-out training splits (in progress).
- **B / B_fair / C / Clate** — prompted log-prob baseline / same with the state prefix shared across queries / cross-encoder / pooled MLP.
- **H1–H7** — decision retention, shared-state economics, probability quality, null + unseen candidates, listwise semantics, KDA memory, workflow utility (IDEA2 §20).

## 9. Doc debt (noticed, not fixed)

1. `REPORT.md` §5 is titled "Pending (phase 4, running)" but every item is done; the leak audit "running" and the
   "Codex two stronger baselines queued" lines are stale (audit done in REVIEW §6; the two baselines were superseded by
   `mcq_lora` / `zs_mcq_8B`, never run as specified). `bench_joint` became `bench_fair`; `joint_24k` was cut (REVIEW §4).
2. `REPORT.md` §3 H2 paragraph still says "Pending: rerun on `joint_v1`" — done (`bench_fair`, §3c).
3. `REPORT.md` §4 scorecard: H1 status says "`joint_v1` pending" (superseded by §3b), there is no H5 row (verdict only
   in §3c/§3d prose), and §4 sits before §3b/§3c/§3d in the file.
4. `REPORT.md` §1 says 16 tests; `tests/test_pipeline.py` has 29. §1 says "H100 ×2"; three pods were used.
5. `REPORT.md` §1/§7 cost lines ($35 / $37) are per-phase; no running total anywhere (≈ $68 by this file's estimate).
6. `REPORT.md` §6 "a second joint seed" is done (`joint_v1_s1`, `joint_emb_s1`).
7. `REVIEW.md` §5 item 8 says `--dump_probs` not implemented; `train.py --dump_logits` now exists (REPORT §3d). The
   utility evaluation itself is still not run.
8. `REVIEW.md` §4 lists IDEA2 wording changes ("reusable neural memory" → "prefix KV cache", H2 "strongly supported" →
   "unmeasured") — `IDEA2.md` §20 still says H2 "already strongly supported by v1" (now actually measured by `bench_fair`).
9. `REVIEW.md` §6 label-name overlap audit (train intents vs Banking77) is still open and not tracked elsewhere.
10. `README.md` run commands are v0: `uv run encode.py` cache build, `train.py --cand_null` (flag is now the default;
    the ablation is `--no_cand_null`), `bench.py --model runs/F/model.pt` (now a run dir). README/RESULTS H1 "holds"
    and H4 K-dependence numbers (0.98 → 0.61 at K=50) are superseded by REPORT §3c/§3d.
11. `PLAN2.md` interface block says `--steps 24000` default; `train.py` default is 12000 (`main_v3` used 24k explicitly).
    PLAN2's "lower 20 frozen / LoRA top 8" is superseded by the revision (tap 20, LoRA 13–20) in the same file.
12. `data.py` docstring points to PLAN2 "Data" for the spec, but the v4 scheme (K-decoupled nulls, cse/ksweep/null
    slices) is only specified in REVIEW §5 item 6 and in `data.py` docstrings.
13. `runs/*/results.json` do not store `args`; run configs are recoverable only from `best.pt` (HF), W&B config, or the
    one-off pod scripts in `/tmp` (reproduced in §3/§4 here). `runs/bench/` has only `bench.log` (no `results.json`).
14. `main_v3` vs `main_v3_s1` is presented as a seed pair in REPORT §3 but differ in epochs (24k vs 12k steps).
15. `costguard.sh` hard-codes the pod id / ssh file of pod 3; `pod_setup.sh` downloads data v3 only (no `v4/`).
