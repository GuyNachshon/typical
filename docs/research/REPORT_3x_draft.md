# REPORT §3x draft — full-file, stratified re-evaluation of §3w (`nc_v3_tap20` vs `nc_v3_tap20_wf`)

Supersedes every W/external number in §3w, per that section's correction note. Both checkpoints,
every row of every eval file (no `--limit`), scored with the batched `scripts/eval_wf.py` (native,
`max_state=4096`) on one H100 NVL pod (`pcdm-eval`, id `y2izde9hckdg4k`, ~9 min total, deleted after
upload). Throughput: 63,932 rows/checkpoint in 4m26s / 4m23s ≈ **241 rows/s** (target was ≥30 rows/s
at K≤10; the batched `run_batch_native` path replaces the old one-row-at-a-time `native_kv_decide`
call, ~80x). `suffix_overflow` is 0/0 on every file for both checkpoints, including
`tree_choice_cap` (flat K=320) — §3w's "125/500 rows overflow the 1,024-token suffix and are scored
as chance" no longer applies; those rows now go through `run_batch_native`'s hierarchical chunking
instead of `native_kv_decide`'s hard suffix-length assert. `data_wf/eval/wf_rubric_flip.jsonl` was
also regenerated (`scripts/workflow_corpus.py`'s `pair_shuffle`) so its 1,031 pairs interleave all 3
held-out families instead of being family-blocked; uploaded to `guychuk/pcdm-data:wf/eval`.

Raw JSON: `runs/nc_v3_tap20/eval_wf_full.json`, `runs/nc_v3_tap20_wf/eval_wf_full.json` (also on
`guychuk/pcdm-runs`).

## 1. Every eval file, full rows, base vs `_wf`

| file | n | acc base | acc wf | nll base | nll wf | brier base | brier wf | majority-class floor | always-yes floor (noul) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wf_heldout_choice (tool_select) | 1500 | 0.894 | 0.989 | 0.432 | 0.040 | 0.187 | 0.018 | 0.213 | – |
| wf_heldout_noul (eligibility) | 1500 | 0.555 | 0.699 | 0.808 | 0.786 | 0.554 | 0.450 | 0.583 | 0.417 |
| wf_heldout_score (urgency) | 1502 | 0.404 | 0.498 | 1.236 | 2.094 | 0.678 | 0.808 | 0.355 | – |
| wf_heldout_style (12 trained families, held-out rubric) | 1627 | 0.559 | 0.901 | 1.026 | 0.248 | 0.572 | 0.142 | 0.364 | – |
| wf_rubric_flip (3 held-out families) | 2062 | 0.633 | 0.743 | 0.802 | 0.963 | 0.462 | 0.408 | 0.321 | – |
| wf_rubric_shuffled | 3775 | 0.330 | 0.390 | 1.512 | 4.652 | 0.817 | 1.070 | 0.360 | – |
| cua_s1_forms_demo | 196 | 0.179 | 1.000 | 2.765 | 0.000 | 0.972 | 0.000 | 0.270 | – |
| cua_s1_forms_test | 24370 | 0.335 | 0.997 | 2.151 | 0.012 | 0.838 | 0.005 | 0.070 | – |
| jev4b_adversarial | 600 | 0.637 | 0.932 | 0.805 | 0.407 | 0.469 | 0.118 | 0.428 | – |
| jevlogs_triage | 5080 | 0.697 | 0.522 | 0.675 | 0.967 | 0.429 | 0.662 | **0.697** | 0.303 |
| mind2web_choice | 1600 | 0.300 | 0.427 | 1.724 | 1.627 | 0.824 | 0.715 | **0.427** | – |
| pagerduty_trigger | 6000 | 0.776 | 0.779 | 0.723 | 0.627 | 0.424 | 0.333 | **0.792** | – |
| systemone_lite_hard | 5400 | 0.447 | 0.897 | 1.159 | 0.289 | 0.699 | 0.155 | 0.442 | – |
| tree_choice_cap (K=320, was chance) | 720 | 0.506 | 0.539 | 2.113 | 1.810 | 0.765 | 0.633 | 0.114 | – |
| typed_decisions_test (soft gold; acc = argmax agreement) | 2000 | 0.487 | 0.457 | 1.220 | 2.034 | 0.237 | 0.417 | 0.290 | – |
| typed_decisions_train (soft gold) | 6000 | 0.502 | 0.471 | 1.205 | 2.016 | 0.242 | 0.421 | 0.310 | – |

Bold majority-class floors mark files where a model's accuracy is **at or below** the trivial
constant-prediction baseline.

## 2. Per-family breakdown

**`wf_heldout_style`** (12 trained families, held-out rubric wording) — the aggregate +34-point
gain (.559→.901) holds broadly, not from one or two families:

| family | n | acc base | acc wf | nll base | nll wf |
|---|---:|---:|---:|---:|---:|
| relevance | 150 | 0.320 | 0.793 | 1.127 | 0.385 |
| support | 149 | 0.752 | 0.966 | 0.680 | 0.143 |
| adequacy | 150 | 0.447 | 0.940 | 1.438 | 0.163 |
| action_select | 150 | 0.300 | 0.947 | 1.648 | 0.126 |
| completeness | 149 | 0.389 | 0.872 | 1.370 | 0.373 |
| quality | 131 | 0.573 | 0.664 | 0.958 | 0.709 |
| enum_extract | 150 | 0.707 | 0.993 | 0.766 | 0.031 |
| categorical | 127 | 0.724 | 0.913 | 0.785 | 0.245 |
| routing | 132 | 0.667 | 0.894 | 0.906 | 0.384 |
| severity | 79 | 0.494 | 0.987 | 1.201 | 0.022 |
| policy_permit | 149 | 0.557 | 0.973 | 0.929 | 0.074 |
| fact | 111 | 0.874 | 0.874 | 0.353 | 0.300 |

11 of 12 families gain (fact is flat, quality the smallest gain at +9); this is the one W claim
from §3w that full-file re-evaluation confirms cleanly.

**`wf_rubric_flip`** (1,031 pairs, families interleaved by the `pair_shuffle` fix — previously the
file's head was 100% eligibility):

| family | n_pairs | flip rate base | flip rate wf | both-correct base | both-correct wf | acc base | acc wf |
|---|---:|---:|---:|---:|---:|---:|---:|
| eligibility | 328 | 0.308 | 0.521 | 0.235 | 0.500 | 0.581 | 0.739 |
| tool_select | 353 | 0.958 | 0.994 | 0.875 | 0.994 | 0.929 | 0.997 |
| urgency | 350 | 0.557 | 0.629 | 0.160 | 0.191 | 0.410 | 0.490 |
| **overall** | **1031** | **0.615** | **0.720** | **0.429** | **0.565** | **0.642** | **0.743** |

§3w's flip-rate number (.312→.504) was the eligibility row only (1 of 3 families, as the correction
already flagged); the real, all-family flip rate is roughly double that (.615→.720), and the
direction (wf reads the rubric harder than base) holds for all three families.

**`wf_rubric_shuffled`** (own-family query swapped for another row's — a coherent but wrong
rubric; Δ_r = acc(own) − acc(shuffled)):

| family | n | acc base | acc wf | nll base | nll wf |
|---|---:|---:|---:|---:|---:|
| tool_select | 905 | 0.164 | 0.159 | 2.573 | 10.121 |
| eligibility | 1427 | 0.491 | 0.514 | 0.910 | 1.762 |
| urgency | 1443 | 0.276 | 0.411 | 1.441 | 4.080 |

wf's NLL under a wrong rubric roughly doubles-to-quadruples every family (most extreme on
tool_select, 2.6→10.1) while accuracy barely moves — consistent with §3w's "reading the rubric, not
memorizing priors" claim: wf becomes far more confident and far more wrong when the rubric is
swapped, rather than just guessing the same way regardless of rubric.

## 3. Which §3w numbers change sign or lose their footing

- **jevlogs_triage**: §3w reported .162→.832 (flagged in the correction as a yes-rate artifact, not
  accuracy, since the head was 100% `yes`). The full, representative file gives **base 0.697 = the
  exact majority-class floor, wf 0.522 — *below* the floor**. This is not "direction holds, magnitude
  smaller" — it flips: `_wf` is now the *worse* model on this file, and neither number is a real win
  over guessing "no" every time.
- **pagerduty_trigger**: §3w reported .736→.762 (+3, on a head that was 250 `page` + 250 `yes`, zero
  negatives). Full file: base 0.776, wf 0.779 — the delta survives (barely) but **both models sit
  below the majority-class floor (0.792)**. Neither checkpoint beats "always predict the modal
  label" here; the "+3" is real but is a difference between two below-floor numbers, not a signal
  either model does the task.
- **mind2web_choice**: §3w's correction already downgraded this to "+3 over always-`no`"; the
  majority-class floor computed here is **0.427 — identical to wf's own accuracy (0.427)**. wf is
  not beating the constant predictor at all on the full file; base (0.300) is *below* it.
- **tree_choice_cap**: §3w's flat-K=320 rows were "scored as chance" (overflow). With overflow now
  0/0, this file goes from an artifact to a genuine (if modest) result: base 0.506 → wf 0.539, both
  far above the 0.114 majority floor — the one case where fixing the scoring bug turns a discarded
  number into a usable one, in the positive direction.
- **wf_rubric_flip**: the flip rate holds direction (up) but the eligibility-only §3w number (.312→
  .504) understated the real, family-balanced rate (.615→.720) by roughly 2x on both sides — not a
  sign flip, but the effect is larger than §3w could see.
- **typed_decisions / wf_heldout_score / wf_rubric_shuffled**: the calibration-regression direction
  §3w flagged as "robust" (NLL/Brier up under `_wf`) is confirmed at full scale and, if anything,
  larger on `wf_rubric_shuffled` (NLL 1.51→4.65 aggregate, up to 10.1 on tool_select) than the
  train-time pass suggested.
- **cua-s1-forms, jev4b_adversarial, systemone_lite_hard, wf_heldout_choice/noul/score/style**: all
  hold their §3w direction and land within a few points of the previously reported (smaller-sample)
  numbers — these were the files §3w's correction already called "shuffled and representative," and
  the full-file pass confirms that call.

Net: of the 5 external files in §3w's "0–1 of 5 up" line, full-file evaluation now shows
**cua/jev4b/systemone as genuine, majority-beating wins for `_wf`; tree_choice as a newly-recovered
genuine (small) win; and jevlogs/pagerduty/mind2web as at-or-below-floor for both checkpoints** — the
external picture is better than §3w's cautious "0–1 of 5" once tree_choice's overflow bug is fixed,
but jevlogs and pagerduty are worse than either the original §3w numbers or its correction implied:
`_wf` is *below* base and *below* the trivial floor on jevlogs specifically, which the correction
did not anticipate (it flagged the number as unrepresentative, not as sign-reversed).

## 4. Not re-run here

JevBench (`scripts/jevbench_run.py`) is a separate harness (fixed 231/72/36-item public sets, not
`data_wf/eval` or `data_wf_hf/eval` files) and was out of scope for this pass — §3w's JevBench
numbers and their small-n_eff caveats (36 states × 2 paraphrases) stand as previously reported.
