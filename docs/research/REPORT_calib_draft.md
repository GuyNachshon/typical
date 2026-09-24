# REPORT (calibration draft) — S3w calibration experiments, 2026-09-20

Three $0 experiments proposed by reviewers on `nc_v3_tap20` (baseline) and `nc_v3_tap20_wf` (Phase 6A). Local
Mac/MPS was attempted first for everything; the machine was under load average ~220 on 14 cores from
concurrent agent jobs in this session, so experiment 2 and probe 3a were moved to a dedicated pod
(`pcdm-calib`, H100 NVL, deleted after results were pulled). Does not edit REPORT.md; see REPORT.md §3w for
the Phase 6A results this follows up on.

## 1. Null-logit offset knob

Added a scalar `null_offset` (`b`) fitted jointly with `T` by grid search (`T` over the existing
`geomspace(0.1, 10, 60)`, `b` over `[-4, 4]` step `.25`), minimizing NLL on a chosen calibration set
(`--calib_sets`, comma list of eval-set names, default unset = today's exact behaviour with `b` fixed at 0).

`b` is added to `logit(P(∅))` — the *same* operation for both null forms:

- **plain softmax null**: `P(∅) = sigmoid(s_∅/T − logsumexp(s_valid/T))` (a two-outcome softmax between the
  null score and everything else), so adding `b` to the tempered null score before the softmax is exactly
  adding `b` to `logit(P(∅))`, with no change to the relative odds among candidates.
- **factored null**: `r = P(∅)` is composed T-invariantly from `log_r`/`log_1mr`; the same `b` is added
  directly to `logit(r) = log_r − log_1mr`.

`results.json` now stores `T`, `null_offset`, `calib_sets` (`summarize()`'s reporting keys are unchanged).
Default behaviour (`--calib_sets` unset) reproduces today's numbers exactly (`b_grid=None` skips the offset
search and pins `b=0`, bit-identical to the pre-offset `fit_temperature`).

**Tests** (`tests/test_pipeline.py`, all pass):
- `test_probs_from_logits_offset_b_zero_matches_legacy` — `b=0` reproduces the old `probs_from_logits` output
  exactly, for both null forms.
- `test_probs_from_logits_offset_shifts_logit_pnull_by_b` — the identity itself: shifting `b` moves
  `logit(P(∅))` by exactly `b` (to `1e-4`), for both null forms, staying a valid distribution.
- `test_fit_temperature_b_grid_recovers_known_null_miscalibration` — a synthetic set with a true null rate of
  0.85 but a raw null logit that only reaches ~0.28 average `P(∅)` under the best `b=0` fit: letting `b` range
  over the grid recovers `P(∅) ≈ 0.85` (within 0.05) at a meaningfully lower NLL (`Δ > 0.3` nats).
- `test_resolve_calib_examples_default_and_named`, `test_eval_dataset_accepts_T_offset_pairs` — the
  `--calib_sets` name resolution and `eval_dataset`'s `(T, offset)` pair plumbing.

## 2. Refit on soft/held-aside data and re-evaluate

Both checkpoints, `--extra_data data_kb,data_wf,data_wf_hf --eval_cap 1500` (so every set below loads for
both), run on `pcdm-calib` (H100 NVL) against copies of `best.pt` (`runs/{name}_calib_{ctrl,refit}/` — the
originals' `results.json` are untouched):

- (a) control: `--calib_sets val` (v5 val, `b` free).
- (b) refit: `--calib_sets typed_decisions_train,data_wf_val`.

**Data-availability note:** `mmlu_pro` is not present in the current `guychuk/pcdm-data` `kb/` snapshot (only
`mmlu_choicesonly`/`mmlu_cf` were expected there too and are also absent) — the dataset repo has been under
active edit by other PLAN7 tracks this session; the number below is what the live snapshot yields, not a
methodology gap. `TruthfulQA` and `data_kb_val` are reported instead as the K-tier stand-ins that *are*
present.

### `nc_v3_tap20` (baseline)

| | control (`T`=1.040, `b`=0) | refit (`T`=1.796, `b`=**−1.75**) |
|---|---|---|
| CLINC-150 acc / acc_k / false_abstain | .845 / .867 / .053 | .857 / .867 / **.022** |
| TREC-fine acc / acc_k / false_abstain | .440 / .530 / .224 | .504 / .530 / .076 |
| CLINC-OOS acc (= abstain-correctly rate) | .798 | .669 |
| HWU64 acc / acc_k | .769 / .799 | .795 / .799 |
| 20NG acc / acc_k | .521 / .554 | .547 / .554 |
| data_kb_val acc / acc_k | .619 / .600 | (same calib set; not re-shown) |
| TruthfulQA-MC1 acc | .283 | .283 |
| held-out score acc/NLL/Brier/ECE | .402/1.237/.678/.030 | .402/1.265/.689/.055 |
| held-out noul acc/NLL/Brier/ECE | .556/.799/.549/.148 | .556/.708/.507/.080 |
| held-out choice acc/NLL/Brier/ECE | .894/.443/.191/.109 | .899/.661/.272/.254 |
| typed_decisions_test NLL/Brier | 1.206/.260 | 1.162/.242 |

### `nc_v3_tap20_wf`

| | control (`T`=1.040, `b`=0) | refit (`T`=3.353, `b`=**+3.00**) |
|---|---|---|
| CLINC-150 acc / acc_k / false_abstain | .734 / .822 / .168 | **.092** / .822 / **.907** |
| TREC-fine acc / acc_k / false_abstain | .386 / .514 / .326 | .008 / .514 / .992 |
| CLINC-OOS acc (= abstain-correctly rate) | .903 | .999 |
| HWU64 acc / acc_k | .736 / .768 | .055 / .768 |
| 20NG acc / acc_k | .524 / .578 | .044 / .578 |
| held-out score acc/NLL/Brier/ECE | .496/2.032/.802/.373 | .496/**1.085**/.600/.135 |
| held-out noul acc/NLL/Brier/ECE | .699/.767/.446/.177 | .699/.561/.380/.031 |
| held-out choice acc/NLL/Brier/ECE | .989/.041/.018/.006 | .991/.122/.041/.086 |
| typed_decisions_test NLL/Brier | 1.946/.435 | **1.194**/.252 |

Reviewer predictions vs actual (wf model, refit): CLINC-150 false-abstain .168 → ~.06 predicted, **.907
actual** (wrong direction, by a lot); CLINC-OOS → ~.80 predicted, **.999 actual** (right direction, overshoots
because CLINC-OOS "accuracy" here is just correct-abstention rate and the model is now abstaining on
everything); held-out score NLL 2.07 → ~1.3 predicted, **1.085 actual** (better than predicted); typed-decisions
NLL ≤ 1.4 predicted, **1.194 actual** (met).

**Argmax-accuracy note (confirmed, both checkpoints, every set): `acc_k` is bit-identical between control and
refit.** `T`/`b` never touch the ranking among real candidates — CLINC-150 acc_k .8224/.8224, TREC .514/.514,
HWU64 .7677/.7677, 20NG .5784/.5784, held-out score/noul/choice .496/.699/.994 unchanged in both wf rows
above (baseline same pattern). Every accuracy delta in the tables is entirely the ∅-vs-argmax(candidates)
comparison, exactly as the offset is designed to.

**The refit is a domain-transfer failure for the wf checkpoint, not the fix the reviewers expected.** The
joint (T, b) grid search correctly minimizes NLL *on* `typed_decisions_train + data_wf_val` — and it does
genuinely help the soft-target sets it was fit on (held-out score NLL 2.03→1.09, typed-decisions NLL 1.95→1.19,
both beating the reviewers' predictions) — but the single global `b=+3.0` it finds pushes the null threshold so
far toward abstention that CLINC-150/TREC-fine/HWU64/20NG collapse to 5-9% coverage (false-abstain .91-.99).
The *baseline* checkpoint's refit on the identical calibration sets moves `b` the *opposite* direction
(`b=−1.75`, less abstention, CLINC-150 false-abstain actually improves .053→.022) — the same calibration
recipe produces opposite corrections on the two checkpoints, because wf's raw null column is calibrated very
differently against workflow-flavored soft data than against v5 val. This is the sharpest demonstration yet of
REPORT §3w's own caution: **a single global null threshold cannot serve both the intent-classification
label spaces and the workflow/soft-target label spaces at once** — refitting on one family's held-aside data
doesn't transfer, it actively breaks the other family. The fix implied is per-family (or per-set) calibration,
not a bigger/better single `b`.

## 3. Two $0 probes

### 3a. CLINC-150 + catch-all `other`

`scripts/probe_clinc_other.py`, full CLINC-150 test set (4,470 rows), wf model, wf's own fitted `T`=1.040,
`b`=0 (raw calibration, matching REPORT's cached numbers as a sanity check: plain acc here .7338 vs REPORT's
cached .734 — matches).

| | acc | mean P(∅) | mean P(other) |
|---|---|---|---|
| plain (no `other`) | .7338 | .2053 | — |
| + `other` candidate | .7477 | .1882 | **.0005** |

**Mass does not move from ∅ to `other`.** Adding a rendered `other` option barely registers (`P(other)` mean
0.0005 — essentially zero mass), `P(∅)` drops only slightly (.205→.188, −.017), and accuracy actually rises a
little (+.014) rather than falling. This is evidence *against* the simplest version of the E-regression
mechanism REPORT §3w hypothesized (W's rendered catch-alls training a competitor to ∅ on these label spaces):
if the model generically treated any catch-all-shaped candidate as a null-competitor, appending the literal
word `other` here should have pulled meaningful mass off ∅; it does not. The regression is more likely
specific to the *trained* catch-all vocabulary/context (`other`/`not_stated`/`skip` inside W's own rendered
option sets) than a generic "any catch-all word" effect transferable to an unrelated domain like CLINC-150.

### 3b. JevBench hard-tier rubric shuffle (native, wf model)

`scripts/probe_jev_shuffle.py`: the 36 JevBench hard-tier public items with state ≤ 256 tokens (the trained
native length — no truncation confound), identified from `runs/jev_native_v3t20_wf/hard/results.jsonl`. Each
item's `question["criteria"]` (the rubric text `pcdm_jev.decider.query_text` renders) is replaced with another
hard-tier item's criteria of the *same question type* (`choice`/`noul`/`score` — criteria's data shape is
type-specific, so this is also the only structurally valid swap); labels, state, gold and instructions are
untouched. Donor choice is seeded and deterministic (excludes self). Re-decided (not read from the cache) so
"own rubric" here is a fresh, comparable control, not the cached run.

| | n | own-rubric acc | shuffled-rubric acc | chance acc | cached hard acc (same ids) |
|---|---|---|---|---|---|
| `nc_v3_tap20_wf` (native) | 36 | 0.611 | 0.611 | 0.384 | 0.639 |

Per type: choice (n=15) own .667 / shuffled .600; noul (n=20) own .600 / shuffled .650; score (n=1) own 0 /
shuffled 0 (single item, not informative alone).

**Shuffled-rubric accuracy is statistically indistinguishable from own-rubric accuracy** (0.611 vs 0.611 on
the identical 36 items), and both sit well above chance (0.384). This is the opposite of the reviewer's
prediction (own .64 → chance under a shuffled rubric) — on this slice the wf model's hard-tier decisions do
not measurably depend on the rubric's *content* at the length it was trained on.

## Conclusions

1. **The null-offset knob is real and does exactly what it says.** It reproduces old numbers exactly at
   `b=0`, the `b`-shifts-`logit(P(∅))` identity holds algebraically and numerically for both null forms, and
   it recovers a synthetically planted miscalibration that `T` alone cannot fix — `T` and `b` close different
   gaps (scale vs. threshold).
2. **Refitting the null threshold on soft/held-aside workflow data helps exactly the sets it's fit on and
   breaks everything else for the wf checkpoint.** typed-decisions NLL and held-out score NLL both improve
   past the reviewers' predictions (1.19 and 1.09 vs. predicted ≤1.4 and ~1.3), but CLINC-150/TREC-fine/
   HWU64/20NG collapse to 5-9% coverage — the opposite of the reviewers' false-abstain-improves prediction.
3. **The same calibration recipe pushes the two checkpoints' null offsets in opposite directions**
   (baseline `b=−1.75`, less abstention and a genuine CLINC-150 false-abstain improvement; wf `b=+3.0`, near-total
   abstention outside the workflow domain) — a single global threshold cannot serve intent-classification and
   workflow/soft-target label spaces at once; this is the sharpest evidence yet for REPORT §3w's per-family
   calibration recommendation over a bigger/better single knob.
4. **CLINC-150's null column does not treat an appended `other` candidate as a competitor.** `P(other)` stays
   at ~0.0005 mean and `P(∅)` moves only marginally (−.017) — the E-regression mechanism is likely specific to
   W's *trained* catch-all vocabulary/context, not a generic "any catch-all-shaped option pulls mass off ∅"
   effect that would show up on an unrelated domain.
5. **The wf model does not execute JevBench hard-tier rubrics at trained length, at least on this 36-item
   slice.** Swapping in a same-type but semantically wrong rubric leaves accuracy exactly where it was
   (.611 → .611, both far above the .384 chance floor) — evidence against "rubric execution" as the mechanism
   behind the model's hard-tier accuracy on short-state items.
