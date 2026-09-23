# PCDM / Typical — project entry point (2026-09-22)

Read this first. It indexes every document, file, run and finding of the project so far; it does not repeat their
contents. Numbers quoted here are copied from `REPORT.md` (the consolidated results doc), `releases/*.md` and
`runs/*/results.json` / `guychuk/pcdm-runs` artefacts.

## 1. What Typical (PCDM) is, and where it stands

Typical (internal name PCDM, Parallel Calibrated Decision Model) turns a pretrained LM into a direct probabilistic
decision engine `F(x, q, A) → P(y | x, q, A)`, `y ∈ A ∪ {∅}`: an unstructured state `x` is encoded once (KV-cached),
then many natural-language queries `q_i` with runtime-defined candidate sets `A_i` are answered in parallel as
calibrated distributions with an explicit "none of the above" (∅) — no generation, no JSON parsing, no per-query
re-encoding of the state. Three typed primitives sit on top of one native contextual readout: **Choice** (K-way,
factored null), **Noul** (yes/no, a dedicated Bernoulli head routed per row), **Score** (ordinal levels, K-way with
ordinal-smoothed targets, τ = 0.7).

**Public today (`releases/*.md`, HF org `OzLabs`):**

| release | backbone | tap | states | headline (public-subset JevBench) | status |
|---|---|---|---|---|---|
| `typical-small-preview` | Qwen3-1.7B-Base | 20/28 | 256 | std .750 / hard .387 | frozen Phase-6A checkpoint, first public release |
| `typical-small` | Qwen3-1.7B-Base | 20/28 | 1,024 | std .694 / hard .432 | Release 1 (`ts1b`); trades ~1 SE std for calibration + external floors |
| `typical-medium` | Qwen3-4B-Base | 26/36 | 1,024 | std .806 / hard .423 | Release 1 (`tm1b`); the capability/ms knee |

Each ships the self-contained `inference/` package (own `Typical` class, no training-repo dependency) and is
browsable through the local Gradio demo (`demo/app.py`). Training code is not yet public; the release cards
document the exact recipe and args.

**Frozen/trained ladder, same recipe, two model families (REPORT §3ab, §3ag, §3ah).** Same 6A/Release-1-shaped
recipe run at every size: Qwen3 1.7B/4B/8B/14B (trained + zero-shot frozen controls with 0 and 3 exemplars) and,
from the Qwen3.5 port (commit `9b1ffac`), zero-shot frozen controls at Qwen3.5 2B/4B/9B/14B plus a full
Release-1 retrain at Qwen3.5-4B (`tm2`). Headline: capacity buys standard-tier accuracy, knowledge and
in-distribution workflow decisions roughly monotonically at near-constant latency (JevBench std .750→.833→.875
at 1.7B→4B→14B, 45→56→60 ms/decision); Qwen3-8B is a checkpoint outlier in both frozen and trained form; the hard
tier and probability quality do **not** scale with parameters — they are data/objective problems fixed by the
long-state and calibration work below, not by size. `tm2` (Qwen3.5-4B, Release-1 recipe, tap 23, 2,048-token
states) is the one exception to "training buys standard, costs hard at scale": JevBench std .861 / **hard .495**
— above every Qwen3 model at 4B and within noise of the *frozen* 14B (.559) — making the Qwen3.5 family switch
the leading open question for `typical-medium-v2` and `typical-large`.

**Unreleased / in flight, 2026-09-21/22 (PLAN7 Track A/D, §3ag–§3ah):**
- **The long-state bug and its fix.** `data_wf_long` rendered `Case: <facts> <request>` last; right-truncation at
  `--max_state` silently dropped the facts on 98.8% of those rows at 1,024 tokens (all of them at 256) — every
  model since Phase 6A was trained to answer long policies from unreadable states. Fixed (commit `2fad326`):
  facts-first regeneration, `--drop_truncated` (drop rows that don't fit rather than cut them), `--grad_ckpt`,
  `--best_on` (checkpoint selection on uncertainty+curriculum val NLL), frozen-backbone teacher
  labels (`scripts/teacher_label.py --zero_shot --shots 3`). A second bug in the same window: the SDPA padding
  mask was built `long` instead of `bool`, forcing PyTorch's O(L²) math kernel and causing the 14B's long-state
  OOMs (commit `6d0a7e3`).
- **`tl1b`** — Qwen3-14B with every §3ag fix (facts-first long corpus, 3,072-token states, frozen-14B KD, 8k
  steps, calibration-val checkpoint selection): JevBench std **.931** (best in the project, level with
  `system-one-open`'s public-subset number), hard Brier .85→.66, held-out score NLL 2.87→0.95, long_policy tripled
  (.05→.158), level-7 composition .620 (first model clearly above chance there) — but hard *accuracy* barely moves
  (.468→.450) and still trails the frozen 14B-with-3-shots (.559). Pass rule for `typical-large` (hard ≥ .559 or
  Brier ≤ .65, long_policy ≥ .35) is narrowly missed on Brier (.656) — **not released**.
- **`tl1b_nokd`** — the matched no-KD control for `tl1b`; checkpoint trained, not yet evaluated.
- **`tl2`** — Qwen3-9B (Qwen3.5 generation) on the `tl1b`-shaped recipe; checkpoint trained, not yet evaluated.
- **`ts1c`** — a 1.7B re-run; base checkpoint trained, not yet evaluated. Its tap-depth sweep (`ts1c_tap15/17/18`
  vs the release tap 20) *is* evaluated: no tap in {15,17,18,20} dominates on every axis (JevBench std/hard,
  PagerDuty, level-7 all trade against each other) — "no free win" from moving the tap alone (see §5).
- **`ts1b_semif` / `tl1b_semif`** — planned SemIf-style-render retrains at 1.7B/14B; not started. What exists
  today is the *zero-shot* semif-rendering probe (`jev_zs_q35_{2b,4b,9b}_semif`): frozen Qwen3.5 backbones scored
  with the leaderboard-style rendering instead of ours. It closes most of the standard-tier gap to the published
  leaderboard, and the effect grows with size (Qwen3.5-9B: std .806→**.931**, hard .541→**.595** — the best frozen
  number in the project; 4B: std .764→.847; 2B: std .583→.597) — rendering, not just scale, explains a large share
  of the leaderboard's .83–.99 standard numbers.
- **Serving.** The public `inference/` path was deep-copying the whole prefix KV cache per decision; replaced with
  a stride-0 view + cached masks/position tensors + a rendered-option cap (`runs/serve_bench2/`). Warm p50: 1.7B
  21–25→**15.5–17 ms**, 4B 26–27→**19–21 ms**, Qwen3.5-4B 42–53→34–46 ms (reference DeltaNet kernels on that pod;
  to be re-measured with FLA + causal-conv1d). `torch.compile`/CUDA graphs tried and rejected (stale-buffer crashes
  against the mutable HF cache, Δp up to .1 across shape buckets); manual per-bucket graph capture is the
  remaining path to ~10 ms.

**Release lineage:** `typical-small-preview` (frozen `nc_v3_tap20_wf`) → `ts1`/`tm1` (routing-bug casualties) →
`ts1b`/`tm1b` → `typical-small`/`typical-medium` (frozen, current). `typical-large` candidate `tl1b` exists and is
documented but not released; the family decision (Qwen3 vs Qwen3.5 ladder for `typical-large`/`typical-medium-v2`)
is open (§7).

## 2. Document map (read in this order)

| doc | what it is | read when |
|---|---|---|
| `docs/plan/idea.md` | v0 statement of the thesis: §1 core equation, §2 desired properties, §3–§16 architecture/training stages, §17 H1–H5, §18 baselines, §20 v0 experiment | you want the original motivation and the pre-registered hypotheses |
| `docs/plan/IDEA2.md` | PI's revised statement after v1: §3 "pretrained state–query interaction is load-bearing" (joint prefix architecture), §6 listwise candidates, §8–§12 training stack (Stage A/B/C, RLCD), §14–§19 KDA (deferred), §20 H1–H7, §21 roadmap Phases 1–5 | you want the current architectural position and the full hypothesis stack |
| `docs/plan/PLAN.md` | v0 PoC plan (Mac/MPS, frozen 0.6B): files, JSONL schema, data mix, interfaces, decision rules for H1–H4 | you need the data schema or the v0 rules |
| `docs/plan/PLAN2.md` | v1 plan (RunPod H100): LoRA/backbone decisions, data v2 spec, run schedule, pre-registered rules, local gate, RunPod mechanics, **"Revision after main_s0"** (tap layer, z-score, data v3) | you are about to rent a GPU or change the training config |
| `docs/research/REVIEW.md` | stop-and-rethink review after `joint_v1`: §1 what stands, §2 what does not (baseline fairness, K–null coupling, OOD calibration), §3 the "fancy classifier over Qwen" question → MCQ-LoRA, §4 cuts, §5 ranked plan + §5a cosine probe + §5b outcomes, §6 threats to validity | you want to know *why* Phase 2 ran what it ran and what is still unproven |
| `REPORT.md` | consolidated results: §1 what was built, §2 bugs/findings in order, §3 v1 tower results + H2, §4 scorecard, §3b joint_v1, §3c Phase 2 (MCQ-LoRA, joint_v2/lw/emb, fair bench), §3d joint_emb follow-ups (null bias, emb+lw, seed), §5 pending, §6 next, §7 decision log | you need a number, a verdict, or the reason a decision was taken |
| `docs/research/RESULTS.md` | v0 snapshot table (frozen 0.6B runs B/C/Clate/D/E/F/G, 3 seeds) + v0 H2 bench + v0 verdicts | historical only; superseded by REPORT §3 onwards |
| `README.md` | v0 README: v0 run commands, v0 findings, v0 verdicts | historical only; the CLI it shows is stale (see Doc debt) |
| `docs/research/COMPARE.md` | PCDM vs TypeSafe Jev / System One comparison, refreshed with the JevBench leaderboard table and the frozen-control ladder | competitive framing, before external claims |
| `docs/plan/PLAN7.md` | current roadmap: scaling ladder (Track A) + mixture sweep (Track B) + typed primitives (Track C) + DecisionMix v2 / hard curriculum / U corpus (Track D) + calibration (Track E) + large-K path (Track F); "Execution notes" and "Track D build notes" sections record what each worker actually built and when | you need a track's pass rule, a worker assignment, or the DecisionMix v2 corpus stats |
| `releases/typical-small-preview.md`, `releases/typical-small.md`, `releases/typical-medium.md`, `releases/MANIFEST.json` | public release cards (backbone/adaptation/readout, training args, full results tables, known limitations, license, how-to-run) and the file-copy provenance manifest for each HF upload | before citing a release's numbers or repro command anywhere else |
| `inference/README.md` (+ `inference/typical/`, `inference/example.py`, `inference/test_parity.py`) | the public, training-repo-independent inference package: `Typical.from_pretrained`, `choice`/`noul`/`score`/`decide`, parity verification | integrating the model outside this repo, or checking what the inference port does/doesn't cover (no `n2`/`n2n3` heads) |
| `demo/README.md` (+ `demo/app.py`) | local Gradio app: playground + batch (single-KV-encode) + static release-page tables; `uv run --no-sync python demo/app.py` | showing the model running, or as the release-page content source |
| `docs/research/REPORT_3x_draft.md`, `docs/research/REPORT_calib_draft.md`, `docs/research/REPORT_6b_draft.md` | working drafts behind `REPORT.md` §3x (stratified full-file re-eval), §3y (calibration $0 experiments) and §3ac (Track C typed primitives) — more raw detail than the consolidated section | when a §3x/§3y/§3ac number needs its per-file derivation |
| `/tmp/COMMON_POD_BRIEF.md` | the living ops brief every pod-facing worker reads first: exact Release-1 recipe flags, the standard post-training eval chain, and the Qwen3.5-era addenda (torch/kernel rules, micro-batch table, watcher rules) — §6 below is this file's rules folded into the permanent doc | before touching a pod |
| `docs/plan/PROJECT.md` | this file | first |

## 3. Code map

All Python runs through `uv` (`uv run ...`; on pods `uv run --no-sync ...`). Tests: `tests/test_pipeline.py` (29 tests,
CPU, tiny random-weight Qwen3 from `tests/conftest.py`).

| file | role | key entry points / flags | covered by |
|---|---|---|---|
| `pcdm/encode.py` | backbone + LoRA + caches. `Backbone(name, lora_layers, lora_r, tap_layer)` (hand-rolled `LoRALinear`, `tap_layer` truncates the model, `frozen_features` from `split_layer = tap − lora_layers`), `FeatureCache` (token features per string), `EmbedEncoder`/`VecCache` (Qwen3-Embedding-0.6B, `EMBED_DIM=1024`), `build_cache`, `build_vec_cache`, `pick_device`. **Qwen3.5 port (commit `9b1ffac`, §3ag):** unwraps the VL-style `text_config`/`.language_model` config, handles the hybrid stack (3× Gated-DeltaNet + 1× full attention; a 71%-depth tap keeps 4 of 6 full-attention layers at 0.8B), routes LoRA to `in_proj_qkv/z/b/a`+`out_proj` on DeltaNet layers vs q/k/v/o on attention layers (MLP everywhere), and patches a transformers-5.17 cache gap (`LinearAttentionLayer` lacking `batch_repeat_interleave`) in `native_kv_decide`; cached-vs-full parity 1e-7, Qwen3 behaviour unchanged (153 tests) | `uv run pcdm/encode.py` (v0 cache build); `python -c 'import encode; encode.selftest()'` | `test_lora_zero_init_and_grads`, `test_joint_causal_invariance`, `test_embed_encoder_unit_norm` |
| `pcdm/model.py` | `DecisionModel` (optional cross-attn tower `num_layers`, hybrid frozen-space sim features, candidate-aware null, optional `SetMixer` listwise block, `d_cand` for embedder candidates), `decision_loss` (soft-CE over `[(1−p∅)·target, p∅]`), `collate`, `run_batch(joint=…)`, `split_joint`, `decide` (encode state once, chunk M queries, KV cache) | library only | `test_collate_ragged`, `test_permutation_equivariance`, `test_loss_matches_hand_soft_ce`, `test_null_target_all_mass_on_null`, `test_run_batch_end_to_end`, `test_decide_matches_batched`, `test_joint_run_batch_shapes`, `test_decide_joint_matches_run_batch`, `test_candidate_permutation_equivariance`, `test_padding_invariance`, `test_duplicate_slots_identical`, `test_listwise_identity_at_init_and_warm_start`, `test_add_candidate_odds`, `test_decision_model_d_cand_forward_backward`, `test_run_batch_with_fake_vec_cache` |
| `pcdm/train.py` | train + eval + temperature fit + `results.json` + W&B + HF upload; auto-resume from `last.pt`; `--eval_only` rebuilds the architecture from `best.pt`'s saved args | `--joint --tower_layers 0 --tap_layer 20 --zscore --lora_r 16 --cand_encoder qwen3emb --listwise --readout energy\|mcq\|native --zero_shot --init_from --eval_only --dump_logits DIR --smoke --data DIR --steps 12000 --bs 64 --seed`. **PLAN7-era flags** (all additive, default = unchanged behaviour): `--family_weights E:0.40,K:0.15,W:0.35,U:0.10` + `--bucket_map` (family-balanced sampler over `meta.fam_bucket`), `--null_aug W:0.20` (seeded fraction of hard-labelled W rows get a gold-removed ∅-only copy), `--max_state 1024` (native decision-state truncation length) + `--drop_truncated` (drop rows that overflow it instead of right-truncating — the long-state-bug fix), `--grad_ckpt` (backbone gradient checkpointing for long states), `--best_on data_u_val,data_wh_val` (checkpoint selection on named eval sets' mean NLL instead of in-distribution val), `--brier_lambda` (add λ·Brier to `decision_loss`), `--calib_sets` (joint grid-search fit of T + null-offset b on named sets), `--nc_render semif` (SemIf-structured chat-template render, alongside `letters`/`tags`/`letters_nonull`), `--score_head cumlink` (cumulative-link ordinal head vs default `choice`), `--noul_head bern` (Bernoulli `P(yes)=σ(w·h_D)` vs default 2-way `choice`), `--qtype_filter choice\|score\|noul` (train/val row filter by `meta.qtype`), `--ordinal_smooth 0.7` (ordinal-smoothed Score targets, τ), `--ckpt_upload` (push `last.pt`/`best.pt` to `--hf_repo` every `--ckpt_every`, for pre-emptible pods), `--shots N` (`--readout mcq`: prepend N fixed exemplars so a base model uses the letter format — the frozen-with-shots controls in §3ab/§3ag). | `test_checkpoint_resume_reproduces_loss`, `test_wandb_offline`, `test_null_bias_real_path_on_mini_v4` |
| `pcdm/native.py` | `NativeHead` (factored null, per-row Bernoulli noul routing, `letters`/`tags`/`letters_nonull`/`semif` renderers, `cumlink` score head) + `native_kv_decide` (encode state once into a KV cache, score every query's suffix against it) — the module the public `inference/typical/native.py` is a trimmed port of | library only, via `pcdm/train.py --readout native` | — |
| `pcdm_jev/decider.py`, `pcdm_jev/adapter.py` | `PCDMDecider` — the harness-facing `decide()` used by every JevBench run and the `inference/` package's numerically-identical model; `mode="mcq_zero_shot"` reads next-token letter logits over rendered options from a frozen backbone (no checkpoint) — the zero-/few-shot frozen-control path used at every ladder size in §3ab/§3ag | `PCDMDecider(run_dir, mode="native"\|"mcq_zero_shot", shots=N)` | — |
| `pcdm/data.py` | HF datasets → JSONL (`train/val/eval/*.jsonl`, `held_out_intents.json`); v4 scheme = K-decoupled nulls + `cse_*`/`ksweep_*`/`null_irrq_*`/`null_nearmiss_*` eval sets; `TRAIN_NAMES`/`PARAPHRASES`; `selftest()` leak audit; **v5 being added by `w-datav5`** | `uv run pcdm/data.py [--small] [--out DIR]`, `uv run pcdm/data.py --selftest --out data_v4` | `test_data_audit_if_present` |
| `pcdm/mcq.py` | `MCQHead` = same backbone/LoRA, answer read from next-token letter logits over enumerated options + "none" letter; chunked two-stage above 51 options | via `pcdm/train.py --readout mcq` | `test_mcq_letter_logits_shape_finite`, `test_mcq_padded_columns_are_finfo_min`, `test_mcq_chunked_k70` |
| `pcdm/metrics.py` | `summarize` (acc, acc_k, nll, brier, jsd, ece, auroc_null, auroc_conf, sel_acc80, conf_wrong), `choice_set_effects` (cse battery), `ksweep` (P(∅\|absent) vs K, range) | library | `test_metrics_summarize_hand_values` |
| `pcdm/baselines.py` | B = prompted log-prob with KV-cached prefix (`--backbone`, `--limit`, `--check`); C = LoRA cross-encoder + per-task heads | `uv run pcdm/baselines.py B --backbone Qwen/Qwen3-8B-Base --data data_v3 --name B_8B`; `uv run pcdm/baselines.py C --name C_lora` | none (`--check` self-check only) |
| `pcdm/bench.py` | H2 microbench: ours vs B vs B_fair (prefix shared), K ∈ {4,32,150[,1000]} × M ∈ {1,8,64,256}; writes `bench.json` + `results.json` | `uv run pcdm/bench.py --model runs/joint_v1 --name bench_fair [--full] [--quick]` | none |
| `pcdm/report.py` | side-by-side table of `runs/*/results.json` on the hypothesis columns | `uv run pcdm/report.py [--raw]` | `test_report_handles_na` |
| `scripts/cosine_probe.py` | $0 probe: untrained nearest-label cosine on held-out label spaces for 3 candidate encoders (REVIEW §5a) | `uv run scripts/cosine_probe.py [--limit N]`, `--selftest` | — |
| `scripts/leak_audit.py` | MinHash char-5-gram near-dup audit train vs every eval set → `runs/leak_audit.json` (+ per-set near-dup lists) | `uv run scripts/leak_audit.py` | — |
| `scripts/null_bias.py` | post-hoc `s∅' = s∅ + α·log K + β` fitted with T on val from a `--dump_logits` dir (REPORT §3d(1)) | `uv run scripts/null_bias.py runs/dump_joint_emb --out runs/joint_emb_nullbias/results.json` | `test_fit_null_bias_finite_and_improves_val_nll`, `test_fit_null_bias_three_ragged_sets_end_to_end` |
| `scripts/eval_wf.py` | post-hoc full-state-length evaluation of a checkpoint on any `--files` glob (train-time evals truncate/cap); batched, stratified `--limit`, per-family/qtype/label-class breakdowns, majority baselines. Rebuilt batched+stratified for §3x (superseded the first-500-rows §3w numbers, see the preview card's Errata) | `uv run python scripts/eval_wf.py --run runs/<name> --mode native --files data_wf/eval/*.jsonl data_wh/eval/*.jsonl data_u/eval/*.jsonl --limit 0 --out runs/<name>/eval_wf.json` | — |
| `scripts/decisionmix_v2.py` | PLAN7 Track D: `data_wh` (hard rule-engine curriculum, 12 domains × 6 boolean conditions, levels 1–7, 100% counterfactual rubric groups) + `data_u` (soft-target uncertainty corpus: UNLI-val, `metaeval/ambient`, `metaeval/chaos-mnli-ambiguity` eval-only, four synthetic closed-form generators); reuses `workflow_corpus.py`'s `flip_pairs`/`shuffled_rubric`/`leak_check` unmodified. Full build notes in `docs/plan/PLAN7.md`'s "Track D" section | `uv run scripts/decisionmix_v2.py --corpus wh\|u\|both [--limit N]` | `tests/test_decisionmix_v2.py` (10 tests) |
| `scripts/workflow_corpus.py` | rubric-conditioned typed-workflow generator (`data_wf`); `--long_share TARGET` (PLAN7 Track B) raises the combined `train.jsonl`+`train_long.jsonl` long-state (1–2.6k-token) row share to TARGET, used for the long-state capacity test (`long_e45`) and later the facts-first `train_long_v2.jsonl` regeneration (§3ag fix) | `uv run scripts/workflow_corpus.py --long_share 0.25` | — |
| `scripts/jev_hf_datasets.py` | streams Jev-shaped HF datasets (`cua-s1-forms`, `systemone-lite-general`, `jev-4b-distill-data`) into the training row schema → `data_wf_hf/{train,eval}` | `uv run scripts/jev_hf_datasets.py [--limit N]` | — |
| `scripts/gate_experts.py` | PLAN6 item 3: learned per-input expert gate over the energy/native score mixture — tested, **rejected** (a global-g mixture captures almost none of the oracle envelope; REPORT §3u) | `uv run scripts/gate_experts.py` | — |
| `scripts/teacher_label.py` | soft labels from a teacher onto a jsonl file; `--zero_shot --backbone ... --shots N` scores with a fully frozen backbone + N fixed exemplars instead of an mcq-readout checkpoint — the frozen-teacher-KD label source for `tl1b` (§3ag/§3ah) and the frozen-with-shots controls (§3ab) | `uv run scripts/teacher_label.py --zero_shot --backbone Qwen/Qwen3-14B-Base --shots 3 --file data_wf_long/train.jsonl` | — |
| `inference/typical/{backbone,native,core}.py` | the public, training-repo-independent port of the serving path: `backbone.py` (frozen truncated Qwen3/Qwen3.5 trunk + LoRA), `pcdm/native.py` (`NativeHead` + `native_kv_decide`, same renderers/heads as the training module minus `n2`/`n2n3`), `core.py` (`Typical.from_pretrained` + `choice`/`noul`/`score`/`decide`) | `from typical import Typical; Typical.from_pretrained("OzLabs/typical-small")` | `inference/test_parity.py` (`RUN_SLOW=1`; max abs prob diff 0.0 vs `PCDMDecider` on CPU/MPS) |
| `demo/app.py` | local Gradio release page: model selector (lazy-load, one resident model), Playground (single decision, probability bar chart incl. ∅), Batch (`native_kv_decide` single-KV-encode over several queries), static release-page tables from `releases/*.md` + REPORT §3ab's latency ladder | `uv run --no-sync python demo/app.py` → http://127.0.0.1:7860 | — |
| `tests/conftest.py` | tiny Qwen3 backbone / cache fixtures | — | — |
| `scripts/costguard.sh` | every 10 min: stop a RunPod pod idle (no train/baselines/bench process, GPU < 5 %, no waiter) for 2 checks; pod ids hard-coded in `PODS` | `./scripts/costguard.sh &` (log `/tmp/costguard.log`) | — |
| `scripts/watchdog.sh` | v0 Mac helper: rerun a command if its log stops growing for 5 min (MPS stream hang) | `./scripts/watchdog.sh LOG CMD...` | — |
| `scripts/run_gpu.sh` / `scripts/run_gpu2.sh` / `scripts/run_gpu3.sh` / `scripts/run_gpu3b.sh` | idempotent pod schedules for phases 1 / diag / 3 (pod 1) / 3 (pod 2); skip a run if `runs/<name>/results.json` exists; `timeout Nh` + `runpodctl pod stop` | `tmux new -d 'bash scripts/run_gpu3.sh'` on the pod | — |
| `scripts/run_rest.sh` | v0 Mac schedule (B, cache, F/G/D/E eval, bench, seeds) | historical | — |
| `scripts/pod_setup.sh` | fresh-pod bootstrap: env dirs on `/workspace`, uv, tmux, `hf download` data, 300-step GPU smoke | `bash scripts/pod_setup.sh` | — |

Phase-2 schedules (joint_v1, tower_big, joint_emb_lw, joint_emb_s1, dump) were one-off scripts sent to the pods
(`/tmp/pod{1,2,3}_*.sh` on the Mac, not in the repo); their exact commands are reproduced below.

### Exact commands

```bash
# tests (CPU, < 60 s)
uv run pytest tests/ -q

# data: current pcdm/data.py builds the v4 scheme (33 eval sets). v3 = 22 sets (data/), v4 = data_v4/
uv run pcdm/data.py --out data_v4 && uv run pcdm/data.py --selftest --out data_v4
uv run pcdm/data.py --small                          # 200/source -> data_small/ (local gate)
uv run hf download guychuk/pcdm-data --repo-type dataset --local-dir data_v3          # v3 at repo root
uv run hf download guychuk/pcdm-data --repo-type dataset --include 'v4/*' --local-dir .   # v4 under v4/
uv run scripts/leak_audit.py                    # -> runs/leak_audit.json

# train the current default config (= runs/joint_emb_lw; drop --listwise for joint_emb, add --seed 1 for joint_emb_s1)
uv run python pcdm/train.py --name joint_emb_lw --joint --tower_layers 0 --tap_layer 20 --zscore --lora_r 16 \
    --data data_v4 --cand_encoder qwen3emb --listwise --eval_every 6000 --wandb --hf_repo guychuk/pcdm-runs

# eval-only from best.pt (architecture flags are read from the checkpoint) + logits dump
uv run python pcdm/train.py --name joint_emb --eval_only --data data_v4 --dump_logits runs/dump_joint_emb

# fit the K-aware null bias on the dump (no retraining) -> results.json in pcdm/train.py's schema
uv run scripts/null_bias.py runs/dump_joint_emb --out runs/joint_emb_nullbias/results.json

# side-by-side table
uv run pcdm/report.py            # scaled (T fitted on val);  --raw for unscaled

# fair H2 bench (ours vs B vs B_fair with the state prefix shared; --full adds K=1000)
uv run pcdm/bench.py --model runs/joint_v1 --backbone Qwen/Qwen3-1.7B-Base --name bench_fair --full

# baselines
uv run pcdm/baselines.py B --backbone Qwen/Qwen3-1.7B-Base --data data_v3 --name B_1.7B
uv run pcdm/baselines.py B --backbone Qwen/Qwen3-8B-Base   --data data_v3 --name B_8B
uv run pcdm/baselines.py C --name C_lora
uv run python pcdm/train.py --name zs_mcq_8B --zero_shot --readout mcq --backbone Qwen/Qwen3-8B-Base --lora_r 0 --data data_v3
uv run scripts/cosine_probe.py
```

Smoke / mini gates before renting: `uv run pcdm/train.py --name smoke --smoke --wandb`, then
`uv run pcdm/train.py --name mini_v4 --steps 300 --bs 16 --backbone Qwen/Qwen3-0.6B-Base --data data_small_v4`.

## 4. Run registry

Every `runs/<name>/results.json` (local); `best.pt` for GPU runs lives only on HF `guychuk/pcdm-runs` (private) and
the pod volume. W&B: `guy-na8/pcdm`. "REPORT" = section of `REPORT.md`. Data: v1 = v0 Mac mix (44k), v2 = PLAN2 mix
(758k), v3 = v2 with fixed null synthesis (PLAN2 revision), v4 = K-decoupled nulls + cse/ksweep/null-slice eval sets.

### v0 — Mac, frozen Qwen3-0.6B-Base layer-20 features, 9M tower (docs/research/RESULTS.md, README.md, REPORT §2.1–2.3, §7)

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

### phase 2 — after docs/research/REVIEW.md (REPORT §3c, §3d; REVIEW §5b)

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

### PLAN7 — scaling ladder, mixture sweep, typed primitives, DecisionMix v2, Release 1 and towards `typical-large` (REPORT §3z–§3ah, `guychuk/pcdm-runs`)

Data: v5 = label-space-held-out training splits; `data_wf_long` = 1,024→3,072-token-state workflow rows; `data_wh`/`data_u`
= DecisionMix v2 (`scripts/decisionmix_v2.py`); "6A" = the frozen-backbone Phase-6A recipe (`nc_v3_tap20_wf`); "r1" = the
Release-1 recipe (typed heads + DecisionMix v2 + null-aug + 1,024-token states).

| run | config | question | outcome |
|---|---|---|---|
| `mix_e40a/b`, `mix_e45`, `mix_e50`, `mix_e45_null` | Track B mixture sweep, E/K/W ∈ {.50/.20/.30 … .40/.25/.35}, one with `--null_aug W:0.20` | Pareto point for the family-weighted sampler | `.40/.25/.35` region ties on E/K, W ≥ 6A; `null_aug` fixes abstention/MMLU where an eval-time null offset can't (REPORT §3z) |
| `long_e45` | `mix_e45` + `--long_share 0.25` (25% long-state rows), 256-token states | is the 1.7B hard tier capacity or length | long rows at 256-token training states can't carry their own facts — the first sighting of the long-state bug, before its root cause (right-truncation) was diagnosed (§3ag) |
| `r1_cand`, `r1_bs16`, `r1_null10` | Track B's levers unioned at full batch (bs 64 vs 16, null-aug 10% vs 20%) | Release-1 mix selection | insensitive to null-aug in [.10,.20]; `r1_cand` (E .45/K .20/W .35, 1,024-token states) fixed as the Release-1 recipe base (§3aa) |
| `r1_dmv2` | `r1_cand` + `data_wh` + `data_u` (DecisionMix v2), E .40/K .20/W .30/U .10 | does the hard curriculum transfer | +34–37 within its own rule grammar (rubric-flip .48→.71), ~.50 (chance) on level-7 composition it never trains; U buys NLL not top-1 (typed-decisions NLL 2.06→1.71); JevBench hard .369→.441 (§3ad) |
| `score_A_kway`, `score_B_smooth`, `score_C_cumlink` | Track C Score arms from `nc_v3_tap20_wf`, 3k steps, qtype-filtered: K-way Choice / ordinal-smoothed (τ=.7) / cumulative-link head | which Score type wins on probability quality at equal decisions | B: held-out score NLL 2.07→1.23, ECE .35→.19, identical per-item JevBench-ordinal predictions to A; C wins JevBench std/ordinal Brier but worse internally (undertrained). **Adopted: `--ordinal_smooth 0.7`** (§3ac) |
| `noul_A_2way`, `noul_B_bern` | Track C Noul arms: 2-way Choice vs Bernoulli `P(yes)=σ(w·h_D)` | which Noul type wins | B wins 6/7 sets, exact reversed-label invariance (0/0 vs up to .55), PagerDuty .602→**.886** (first external floor cleared by a margin). **Adopted: `--noul_head bern`, routed per row** (§3ac) |
| `ts1`, `tm1` | first Release-1 attempt at 1.7B/4B | — | `--noul_head bern` applied globally (every row, not just yes/no rows) collapsed both to chance (JevBench std .389/1.7B); fixed per-row (commit `7b9d520`, 140 tests) |
| **`ts1b`** | Release-1 at 1.7B: r1 mix + DecisionMix v2 + ordinal-smoothed Score + per-row Bernoulli Noul + 1,024-token states | Release 1, 1.7B | held-out score NLL 2.03→1.01, typed-decisions 2.06→1.25, first 1.7B above the PagerDuty/jevlogs floors; JevBench std .750→.694 (~1 SE), hard .387→.432. **Frozen as `typical-small`** (§3ae) |
| **`tm1b`** | same recipe on Qwen3-4B-Base, tap 26/36 | Release 1, 4B | knowledge/NLI held, soft-target NLLs halved again, first model >.50 on level-7 (.544), PagerDuty .838, JevBench std .833→.806/hard .432→.423 (matched control `ladder_4b`, within noise). **Frozen as `typical-medium`** (§3af) |
| `ladder_4b`, `ladder_8b`, `ladder_14b` | 6A-shaped recipe (no typed heads, no DecisionMix v2, 256-token states) at 4B/8B/14B, tap ≈71% depth | is hard = capacity | monotone 1.7B→4B→14B on standard/knowledge/workflow at near-constant ms/decision; **8B is a checkpoint outlier** in both frozen and trained form; long-state families and soft calibration do not scale — frozen 14B+3-shots beats trained 14B on hard (§3ab) |
| `zs_1p7b/4b/8b/14b` (0-shot), `jev_zs3_1p7b/4b/8b/14b` (3-shot) | frozen-backbone letter-logit controls, no training, at every Qwen3 ladder size | separate backbone capacity from training | 0-shot standard climbs .583→.722→.375(broken, all-"A" bug)→.819 with scale; 3-shot fixes the 8B bug (.556) and gives the frozen 14B **hard .559**, still the project's best hard-tier number until `tl1b` (§3ab) |
| `jev_zs3_q35_2b/4b/9b`, `jev_zs3_14b`, `jev_zs3_4b_inst` | 3-shot frozen controls on the Qwen3.5 ladder (2B/4B/9B) + Qwen3-14B/4B-instruct for reference | Qwen3.5 vs Qwen3 frozen capacity | Qwen3.5-4B hard .495 vs Qwen3-4B .414 at the same shot count; Qwen3.5-9B hard **.541**, approaching the frozen Qwen3-14B (.559); instruct-tuning Qwen3-4B changes ~nothing (§3ag) |
| `jev_zs_q35_2b/4b/9b_semif` | the same frozen Qwen3.5 checkpoints scored with SemIf-style rendering instead of ours | does rendering explain the leaderboard's .83–.99 standard | yes, and the effect grows with size: 9B std .806→**.931** (level with `system-one-open`), hard .541→**.595**; 4B .764→.847; 2B .583→.597 — the best frozen numbers in the project, all with zero training (§3ag execution notes) |
| **`tl1b`** | Qwen3-14B, `ladder_14b`'s backbone + every §3ag fix (facts-first long corpus, `--drop_truncated`, 3,072-token states, DecisionMix v2+U, typed heads, frozen-14B KD α·CE+β·KL T=2, `--best_on` calibration-val selection), 8k steps | does fixing long-state + calibration recover the pass rule for `typical-large` | JevBench std **.931** (project best), hard Brier .85→.66, held-out score NLL 2.87→0.95, long_policy .05→.158, level-7 .620; hard *accuracy* barely moves (.468→.450), still below frozen-14B+3-shots (.559). **Misses the pass rule narrowly on Brier (.656); not released** (§3ah) |
| `tl1b_nokd` | `tl1b` minus the frozen-teacher KD term | is the KD term load-bearing for `tl1b`'s gains | checkpoint trained; **not yet evaluated** |
| `tm2` | Qwen3.5-4B-Base, Release-1 recipe, tap 23, 2,048-token states, 8k steps | does the Qwen3.5 generation change the size-vs-hard-tier trade at 4B | JevBench std .861 / **hard .495** (Brier std .22/hard .72), MMLU among-K .429/Δq .194 — hard tier matches/exceeds `tl1b` (14B) at 4B parameter count; CLINC .795 (−5 vs `tm1b`). Strongest evidence yet for switching `typical-medium`/`typical-large`'s backbone family |
| `tl2` | Qwen3-9B (Qwen3.5 generation), `tl1b`-shaped recipe | 9B point on the Qwen3.5-recipe ladder | checkpoint trained; **not yet evaluated** |
| `ts1c` | a 1.7B re-run on the Release-1 recipe (tap 20) | — | checkpoint trained; **not yet evaluated** as a standalone point (its tap variants below are) |
| `ts1c_tap15`, `ts1c_tap17`, `ts1c_tap18` | tap-depth sweep at 1.7B on the `ts1c` recipe (tap 15/17/18 vs the release's tap 20) | is there a better tap than 20 for this recipe | no tap dominates: tap15 best PagerDuty (.715) but worst hard (.324); tap17/18 best hard (.441) but worst PagerDuty (.54–.59); level-7 flat ~.48–.51 at every tap. **No free win from moving the tap alone** (new finding, §5) |
| `ts1b_semif`, `tl1b_semif` | planned SemIf-render retrains at 1.7B/14B | does training with SemIf-style rendering change decisions, not just frozen scoring | **not started** — only the zero-shot semif probe above exists so far |
| `serve_bench`, `serve_bench2` | public `inference/` package warm/cold latency, before/after the KV-cache deep-copy fix | serving-path cost, not in-process training-repo latency | `serve_bench` = pre-fix baseline; `serve_bench2/{before,after}.json` = the stride-0-view fix: 1.7B warm p50 21–25→15.5–17 ms, 4B 26–27→19–21 ms, Qwen3.5-4B 42–53→34–46 ms (reference DeltaNet kernels; §3ag) |

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
20. **Per-row Bernoulli routing, not global** — `--noul_head bern` applied to every row (not just rows whose candidate
    set is exactly `{yes,no}`) collapsed the first Release-1 attempt (`ts1`/`tm1`) to chance. Fixed per-row (commit
    `7b9d520`, 140 tests): a row only takes the Bernoulli path if its labels are exactly `{yes,no}`; every other
    2-way set (`{true,false}`, `{approve,reject}`) scores as ordinary Choice, bit-identical to before. REPORT §3ae.
21. **The long-state truncation bug** — `data_wf_long` rendered `Case: <facts> <request>` with the facts *last*;
    right-truncation at `--max_state` silently dropped the facts on 98.8% of those rows at 1,024 tokens (100% at
    256). Every model since Phase 6A/§3z was trained to answer long policies from states that no longer contained
    them — the direct cause of `long_policy` collapsing to .05 at 14B and .21 at `typical-medium`. Fix: facts-first
    regeneration (`train_long_v2.jsonl`) + `--drop_truncated` (drop rows that don't fit rather than cut them),
    commit `2fad326`. REPORT §3ag.
22. **SDPA bool-mask / O(L²) fallback** — the padding mask for scaled-dot-product attention was built as `long`
    instead of `bool`, which silently forces PyTorch's O(L²) math kernel instead of a fused one — the root cause of
    the 14B's OOMs at long (3,072-token) states and of the ladder's general memory pain at scale. One-line fix
    (commit `6d0a7e3`). REPORT §3ag.
23. **Zero-copy serving fix** — the public `inference/` serving path deep-copied the entire prefix KV cache on
    every decision; replaced with a stride-0 view (bit-identical, zero extra bytes) plus cached masks/position
    tensors and a rendered-option cap. Warm p50: 1.7B 21–25→15.5–17 ms, 4B 26–27→19–21 ms (`runs/serve_bench2/`).
    REPORT §3ag.
24. **CUDA graphs rejected for serving** — `torch.compile`/manual CUDA-graph capture reached ~8 ms on one fixed
    shape, but the probability output shifted up to Δp = .1 across shape buckets and the mutable HF KV cache caused
    stale-buffer crashes on recapture. Not adopted; manual per-bucket graph capture (a fixed pool of shapes,
    recaptured on cache resize) is the remaining path to ~10 ms, not attempted yet. REPORT §3ag.
25. **The tap-depth sweep found no free win** — re-running the Release-1 1.7B recipe at tap 15/17/18 (vs the
    shipped tap 20) shows every tap trades against every other: tap 15 wins PagerDuty (.715) and loses hard (.324);
    tap 17/18 win hard (.441) and lose PagerDuty (.54–.59); level-7 composition is flat at chance (~.48–.51) at
    every tap tried. Moving the tap alone is not a lever for the hard tier or level-7 — consistent with §5 item 21's
    finding that those are data/objective problems, not architecture-depth problems. `ts1c_tap{15,17,18}`, local.
26. **Qwen3.5 beats Qwen3 on hard tier at matched size, at some standard-tier cost.** The Release-1 recipe on
    Qwen3.5-4B-Base (`tm2`, tap 23, 2,048-token states) reaches JevBench hard .495 — above `tl1b`'s 14B (.450) and
    every Qwen3 model up to 14B — while CLINC-150 lands 5 points below `tm1b`. The frozen zero-shot ladder shows
    the same pattern before any training: Qwen3.5-9B's 3-shot hard (.541) already approaches the frozen Qwen3-14B
    (.559), and SemIf-style rendering alone (no training) pushes it to standard .931 / hard .595. The generation,
    not just parameter count, moves the hard tier — though rendering's effect on hard is mixed at 4B (.495→.468).
    REPORT §3ag; `tm2`, `jev_zs3_q35_*`, `jev_zs_q35_*_semif`.
27. **Rendering explains a large share of the leaderboard's standard-tier numbers.** Scoring the same frozen Qwen3.5
    checkpoints with SemIf-style rendering instead of ours closes most of the standard-tier gap to the published
    leaderboard, and the effect scales with model size (2B +1, 4B +8, 9B +13 points standard) — the leaderboard's
    .83–.99 standard entries are not simply "bigger frozen model reads logits," they are also a rendering choice.
    `jev_zs_q35_*_semif` vs `jev_zs3_q35_*`, REPORT §3ag.
28. **A Qwen3.5-2B retrain point does not exist.** Unlike `tm2` (4B) and `tl2` (9B), no Qwen3.5-2B checkpoint or
    eval is on `guychuk/pcdm-runs` — the zero-shot ladder's 2B point is also its weakest by a wide margin (hard
    .387–.441 even with semif rendering, vs 4B's .495–.468 and 9B's .541–.595), consistent with training budget
    going to the two sizes that looked informative rather than to a full 2B/4B/9B/14B retrain grid.

## 6. Ops (current rules, 2026-09-22; source of truth is the living `/tmp/COMMON_POD_BRIEF.md`)

**Repo/git discipline on every pod.** `git pull --no-rebase origin gpu-runpod-full-experiment` first; never
stash/reset/pull --rebase. Build the pod tarball with `git archive HEAD` (not the working tree). Commit only the
paths that pod's worker owns, trailer `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`, push
after every commit. Python via `uv run --no-sync python` everywhere (never bare `uv run`, which auto-syncs and can
corrupt a `.venv` shared across pods).

**Pod-create recipe.** A pod created with `--startSSH` never reaches "ready" on this account and bills anyway (12
stalls, ~$13 lost); `--ports "22/tcp"` boots in ~30 s — use that. Pods are already created and registered with the
cost guard before a worker starts; workers do not create pods. **Stop = wipe** (no volume) — everything not yet
pushed to git or uploaded to HF is gone the moment a pod stops; upload before stopping, not after. Delete the pod
once its chain has uploaded everything, then remove its line from `/tmp/PODS_ACTIVE`.

**Secrets and shell hygiene.** scp `.env`, `chmod 600`, and source it as
`{ set +x; } 2>/dev/null; set -a; . /workspace/.env; set +a; set -x` — never `set -x` across an `.env` source (it
prints every secret to the log). Never `pkill -f` inline over an SSH one-liner (it self-matches the SSH shell and
kills the session); kill from a script file with `pgrep -f` + explicit `kill`. Launch chains with `nohup` from a
script file, one nohup chain per pod for all stages — never leave a stage to an agent hand-off mid-chain.

**torch / driver / kernel rule (Qwen3.5-era addendum).** Keep the locked torch (2.14+cu130) on hosts whose
`nvidia-smi` reports CUDA 13.x. Reinstall a cu128 wheel (`torch==2.11.0+cu128`) **only** when the host driver
itself reports CUDA 12.8 — never "to match a kernel wheel": a stray cu128 reinstall on a CUDA-13 host broke SDPA
backward (`mha_graph.execute … is_good()`) and downgraded triton, which then fails FLA's Hopper check. Once
training has started, do not touch the venv again — if a kernel import fails, fall back to the reference path
instead. Qwen3.5 needs `flash-linear-attention` (triton ≥ 3.7.1 on Hopper) + `causal-conv1d` built against the
exact torch in the venv; **verify with `python -c "import fla, causal_conv1d"`**, not by checking that `pip`/`uv`
reported success.

**Micro-batch by size, 2,048-token states on 80 GB:** 1.7B/2B → `--grad_accum 8` (micro 8); 4B → 8; 9B → 16; 14B
(3,072-token states) → 32. `--eval_bs` ≤ the micro-batch size.

**Watchers.** Every chain gets a watcher on stall (no `step N` line for 20 min) and on `RuntimeError|Traceback|
FATAL|OOM` appearing in **both** the run log and the chain's own stdout log — a crash that only shows up in one of
the two has been missed before.

**Release-1 1.7B recipe** (`ts1b`, REPORT §3ae, reproduced here because it's the thing every new run is diffed
against): `--readout native --nc_head n3 --nc_render letters_nonull --null factored --tap_layer 20 --zscore
--lora_r 16 --lora_layers 8 --data data_v5 --extra_data data_kb,data_wf,data_wf_hf,data_wf_long,data_wh,data_u
--bucket_map data_wf_long=W,data_wh=W,data_u=U --family_weights E:0.40,K:0.15,W:0.35,U:0.10 --null_aug W:0.20
--ordinal_smooth 0.7 --noul_head bern --max_state 1024 --bs 64 --grad_accum 4 --eval_cap 1500 --steps 12000
--ckpt_upload --wandb --hf_repo guychuk/pcdm-runs`. **Updated defaults for new runs** (post-§3ag fixes):
`--max_state 2048 --drop_truncated --grad_ckpt --best_on data_u_val,data_wh_val --steps 8000 --val_every 2000
--eval_every 8000` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Standard post-training chain:** probe (`--name probe_<run> --eval_only --data data_probe`) → JevBench native
(`scripts/jevbench_run.py --jevbench_dir /workspace/jevbench --model runs/<run> --mode native --name <run> --device
cuda`) → `scripts/eval_wf.py --run runs/<run> --mode native --limit 0 --max_state 4096 --files "data_wf/eval/*.jsonl"
"data_wf_hf/eval/*.jsonl" "data_wh/eval/*.jsonl" "data_u/eval/*.jsonl" --out runs/<run>/eval_wf.json` →
`pcdm/bench.py --native` → upload everything under `<run>/`, `probe_<run>/`, `jev_native_<run>/`, `bench_<run>/` to
`guychuk/pcdm-runs` → echo `<RUN>_DONE`.

**Infra (unchanged since v0/v1).** RunPod Secure Cloud H100 SXM (≈ $3/h), 100 GB network volume at `/workspace`
(HF_HOME, UV_CACHE_DIR, data, runs, `.env`), template `runpod/pytorch`. Artifacts: W&B `guy-na8/pcdm`; HF
`guychuk/pcdm-data` (v3 at root, `v4/`/`v5/`/`wf/`/`wh/`/`u/` subdirs); HF `guychuk/pcdm-runs` (every run's
`best.pt` + eval jsons, private) plus the three public `OzLabs/typical-*` repos. Secrets in `.env` (never print).

**Cost guard.** `scripts/costguard.sh` reads pods from `/tmp/PODS_ACTIVE` (workers append `<id>|<sshfile>|<waiter>`) and
stops one idle (no trainer process, GPU util < 5%, no waiter) for 2 checks running every 10 min.

**Older pitfalls, still true** (REPORT §2.9; PLAN2 "RunPod mechanics"; user memory notes `remote-job-hygiene.md`,
`mps-sync-deadlock.md`, `decoder-features-for-heads.md`):
- `tmux kill-session` does not kill `timeout`-wrapped schedule children → ghost schedule trains a second model on
  the same GPU; kill the `timeout`/`bash -c schedule` pids first.
- `runpodctl --env` variables do not reach SSH sessions; source `/workspace/.env` in the script.
- Mac/MPS: boolean indexing / per-item `.item()` hangs under GPU contention; `scripts/watchdog.sh` restarts a stalled log.
- `--eval_only` takes architecture flags from `best.pt`, not the CLI; `--readout mcq` rejects `--zscore/--joint/--listwise`.
- `runs/dump_*/` and `*.pt` are gitignored; `best.pt` for GPU runs is on HF only.

## 7. Open questions and next steps (REPORT §6 "Next" + 2026-09-21/22 findings)

1. **Hard-tier ceiling.** Every fix tried so far (long-state, calibration objective, frozen-teacher KD) moves
   JevBench hard *Brier* (.85→.66 at 14B) far more than hard *accuracy* (.468→.450) — `tl1b` still trails the
   frozen 14B-with-3-shots (.559) on accuracy. Is the remaining gap a data problem (the generator still doesn't
   produce the reasoning shapes hard items need) or an objective problem (accuracy-shaped loss vs a model that is
   "confidently wrong" rather than "uncertain")? No run yet isolates the two.
2. **Level-7 composition** (temporal / units / expected-value / trade-off). Stuck at ~.50 (chance) across every
   model and every tap tried (`r1_dmv2` → `ts1b` → `tm1b` → `tl1b`'s .620 is the first exception, driven by KD +
   calibration, not new data) — the generator has never produced this composition family. Next: extend
   `scripts/decisionmix_v2.py`'s rule engine with explicit temporal/numeric/EV/trade-off composition rows, not more
   of the existing levels 1–6.
3. **Calibration objective, generalized.** `--best_on` calibration-val selection worked at 14B
   (`tl1b`; no Brier term was trained — `--brier_lambda` is 0.0 in every shipped/candidate run); Phase 10's full
   `L = log + λ_B·Brier + λ_O·ordinal` per-type/per-tier objective (no single global T)
   is still open, and has not been run at 1.7B/4B where the calibration collapse was first diagnosed (§3ab).
4. **SemIf-style render, now that it's measured.** The zero-shot probe (`jev_zs_q35_*_semif`) shows rendering alone
   recovers most of the leaderboard's standard-tier gap and grows with size (9B: std .806→.931). `ts1b_semif` /
   `tl1b_semif` (training with this render, not just scoring with it) would settle whether the gain compounds with
   training or is a scoring-time artifact of our own render being worse — not started yet.
5. **The Qwen3.5 family decision.** `tm2` (Qwen3.5-4B, Release-1 recipe) beats every Qwen3 model up to 14B on
   JevBench hard (.495) at 4B parameters, at a 5-point CLINC-150 cost. Does `typical-medium-v2` and `typical-large`
   retrain on Qwen3.5, and does the 2B point get filled in before that decision (see findings-ledger item 28)?
   `tl2` (Qwen3.5-9B on the `tl1b` recipe) is trained but not yet evaluated — its numbers are the next input.
6. **CUDA graphs for serving.** Rejected once (Δp up to .1 across shape buckets, stale-buffer crashes against the
   mutable HF cache) at ~8 ms on a single shape. Manual per-bucket graph capture (fixed pool of shapes, recapture
   on cache resize) is the remaining path from the current 15.5–21 ms warm p50 toward ~10 ms; not attempted.
7. **Large-K path** (energy → top-r → native, PLAN7 Track F). Batched marginal cost at K = 256 grows ~4× from
   1.7B to 14B (28→110 ms/query, apples-to-apples table, §3ab) — this matters more with scale, not less, and no
   run has benchmarked the composed path (only its pieces individually: §3n support gate, §3o fusion, §3p letter-
   free native, all pre-PLAN7).
8. **Training-code release.** All three public models ship inference-only; every release card says "training code
   release to follow." No date is set; blocked on nothing technical, just sequencing against the still-changing
   Release-1 recipe (the long-state and calibration fixes landed *after* `typical-small`/`typical-medium` shipped).
9. **8B tap/lr sweep** — deferred (REPORT §6) until a product need for that exact size appears; the ladder's 8B
   anomaly (checkpoint outlier, both frozen and trained) is diagnosed as a checkpoint issue, not chased further.
10. Closed and not reopened: candidate-blind Z, further letter-readout variants, confidence-only routing, null
    functional forms, depth sweeps as a hard-tier lever (item 25 in the findings ledger settled this one more time).

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
- **energy readout vs MCQ readout** — learned per-candidate score + null logit (ours) vs next-token letter logits over enumerated options (`pcdm/mcq.py`, the "Qwen does everything" baseline).
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
7. `docs/research/REVIEW.md` §5 item 8 says `--dump_probs` not implemented; `pcdm/train.py --dump_logits` now exists (REPORT §3d). The
   utility evaluation itself is still not run.
8. `docs/research/REVIEW.md` §4 lists IDEA2 wording changes ("reusable neural memory" → "prefix KV cache", H2 "strongly supported" →
   "unmeasured") — `docs/plan/IDEA2.md` §20 still says H2 "already strongly supported by v1" (now actually measured by `bench_fair`).
9. `docs/research/REVIEW.md` §6 label-name overlap audit (train intents vs Banking77) is still open and not tracked elsewhere.
10. `README.md` run commands are v0: `uv run pcdm/encode.py` cache build, `pcdm/train.py --cand_null` (flag is now the default;
    the ablation is `--no_cand_null`), `pcdm/bench.py --model runs/F/model.pt` (now a run dir). README/RESULTS H1 "holds"
    and H4 K-dependence numbers (0.98 → 0.61 at K=50) are superseded by REPORT §3c/§3d.
11. `docs/plan/PLAN2.md` interface block says `--steps 24000` default; `pcdm/train.py` default is 12000 (`main_v3` used 24k explicitly).
    PLAN2's "lower 20 frozen / LoRA top 8" is superseded by the revision (tap 20, LoRA 13–20) in the same file.
12. `pcdm/data.py` docstring points to PLAN2 "Data" for the spec, but the v4 scheme (K-decoupled nulls, cse/ksweep/null
    slices) is only specified in REVIEW §5 item 6 and in `pcdm/data.py` docstrings.
13. `runs/*/results.json` do not store `args`; run configs are recoverable only from `best.pt` (HF), W&B config, or the
    one-off pod scripts in `/tmp` (reproduced in §3/§4 here). `runs/bench/` has only `bench.log` (no `results.json`).
14. `main_v3` vs `main_v3_s1` is presented as a seed pair in REPORT §3 but differ in epochs (24k vs 12k steps).
15. `scripts/costguard.sh` hard-codes the pod id / ssh file of pod 3; `scripts/pod_setup.sh` downloads data v3 only (no `v4/`).
16. Item 10 (README.md stale) is resolved by this pass — `README.md` is now a current front page, not the v0
    Mac-PoC document; if it drifts again, the v0 content is recoverable from git history, not from this file.
17. `releases/MANIFEST.json` has provenance entries for `typical-small-preview` and `typical-small` only —
    `typical-medium` (`tm1b`) shipped without a manifest entry; not fixed here (release-provenance format is the
    lead's/release worker's to extend, this pass only touches the `.md` docs).
18. No release card exists yet for `tl1b` (documented in REPORT §3ah and this file's run registry, but not
    released — correctly no card) or for `tm2` (trained and evaluated, not a frozen release decision yet — also
    correctly no card). If either is frozen as a release, `releases/*.md` + `MANIFEST.json` need a new entry each.
19. `docs/plan/PLAN7.md`'s pass rule for `typical-large` (hard ≥ .559 or Brier ≤ .65, long_policy ≥ .35) is stated once at
    the top of §3ag's "In flight" paragraph; REPORT §3ah's verdict ("missed narrowly on Brier, .656") is not
    reflected back into docs/plan/PLAN7.md itself — docs/plan/PLAN7.md is a plan document and this pass leaves it as originally
    written, per the instruction not to rewrite the lead's planning doc, but a reader diffing docs/plan/PLAN7.md against
    REPORT §3ah should know the rule as *evaluated* lives only in REPORT.md.
20. Three checkpoints exist on `guychuk/pcdm-runs` with no `results.json` at all yet: `tl1b_nokd`, `tl2`, `ts1c`
    (base, not the tap-sweep variants, which are evaluated). Every table in this pass marks their cells "–"; there
    is nothing to source them from until their eval chains run.
