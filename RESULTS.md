# PCDM / Typical — results (2026-09-22)

Regenerated as one set of tables per model-family size, Qwen3 and Qwen3.5 side by side. Every number is pulled
from `runs/*/results.json` / `runs/probe_*/results.json` / `runs/jev_native_*/summary.json` /
`runs/bench_*/bench.json` (local, or `guychuk/pcdm-runs` where not pulled locally) — none are copied from prose
summaries or estimated. Missing cells are `–`, not a guess. Narrative and verdicts: `REPORT.md`; plan: `PLAN7.md`;
index: `PROJECT.md`; release cards with full training args: `releases/*.md`.

**How to read this.**
- **JevBench is a public-subset run, not a ranked leaderboard entry** (72 standard / 48 easy / 111 hard of the
  harness's 231 public ids; the 146 judge items are not public). Majority/chance baselines on this split are .311
  standard / .284 easy / .336 hard (n = 72 standard, SE ≈ .058, **n_eff = 36** — treat any standard-tier delta
  under ~2 points and any hard-tier delta under ~6 points between our own checkpoints as noise, not a ranking).
  Reported probabilities are the head's softmax conditioned on non-∅ (P(∅) dropped, renormalized).
- **Release status** distinguishes: *released* (frozen, public HF weights + card), *candidate* (fully evaluated,
  a documented pass/fail verdict, not released), *trained* (checkpoint + eval exist, no release decision made),
  *in flight* (checkpoint exists, not yet evaluated — cells below are `–`), *frozen control* (no training at all —
  a base-model zero-shot/few-shot letter-logit read, `pcdm_jev` `mode="mcq_zero_shot"`).
- **held-out noul/score/style**, **typed-decisions**, **wh/u** columns only exist for trained checkpoints — a
  frozen control has no typed head to evaluate on them, hence "–" for every Qwen3.5-2B/9B and Qwen3-1.7B/4B/8B/14B
  frozen-control row.
- Single-decision latency is in-process on one H100, one decision at a time, model load excluded; the K=2/32/256
  triples below are same-pod, same-torch-build ("apples-to-apples") numbers where available (REPORT §3ab), not the
  mixed-host numbers superseded by that ladder.

## 1. Evidence / knowledge classification (CLINC-150 / TREC-fine / HWU64 / 20NG; SNLI / MNLI / BoolQ / ANLI)

| model | checkpoint | release status | CLINC-150 | TREC-fine | HWU64 | 20NG | SNLI | MNLI | BoolQ | ANLI |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | `ts1b` | released (`typical-small`) | .801 | .508 | .761 | .540 | .894 | .859 | .824 | .497 |
| Qwen3-4B | `tm1b` | released (`typical-medium`) | .847 | .414 | .769 | .588 | .909 | .861 | .843 | .544 |
| Qwen3-8B | `ladder_8b` | trained (6A recipe, no Release-1 retrain) | .769 | .486 | .740 | .452 | .853 | .831 | .822 | .468 |
| Qwen3-14B | `tl1b` | candidate (`typical-large`, not released) | .827 | .508 | .760 | .690 | .911 | .868 | .882 | .583 |
| Qwen3.5-2B | – | frozen control only | – | – | – | – | – | – | – | – |
| Qwen3.5-4B | `tm2` | trained (Release-1 recipe, no release decision) | .795 | .480 | .717 | .579 | .900 | .855 | .840 | .554 |
| Qwen3.5-9B | – | frozen control only (`tl2` trained, not yet evaluated) | – | – | – | – | – | – | – | – |

## 2. Knowledge probe (MMLU-Pro among-K, Δ_q_sh) and TruthfulQA

| model | checkpoint | MMLU-Pro among-K | Δ_q_sh | TruthfulQA MC1 |
|---|---|---|---|---|
| Qwen3-1.7B | `ts1b` | .343 | .127 | .257 |
| Qwen3-4B | `tm1b` | .458 | .193 | .408 |
| Qwen3-8B | `ladder_8b` | .302 | .093 | .277 |
| Qwen3-14B | `tl1b` | .498 | .235 | .481 |
| Qwen3.5-2B | – | – | – | – |
| Qwen3.5-4B | `tm2` | .429 | .194 | .414 |
| Qwen3.5-9B | – | – | – | – |

## 3. Held-out workflow (W), typed-decisions, and external floors — full-row `eval_wf`

| model | checkpoint | held-out noul | held-out score (NLL) | held-out style | typed-decisions acc (NLL) | PagerDuty (floor .792) | jevlogs (.697) | Mind2Web (.427) | tree-choice |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | `ts1b` | .710 | .522 (1.01) | .861 | .479 (1.27) | .560 | .500 | .351 | .469 |
| Qwen3-4B | `tm1b` | .811 | .528 (1.03) | .861 | .540 (1.17) | .553 | .480 | .547 | .683 |
| Qwen3-8B | `ladder_8b` | .681 | .543 (1.67) | .882 | .407 (2.14) | .321 | .501 | .404 | .411 |
| Qwen3-14B | `tl1b` | .831 | .621 (0.95) | .882 | .600 (1.04) | .805 | .514 | .564 | .506 |
| Qwen3.5-2B | – | – | – | – | – | – | – | – | – |
| Qwen3.5-4B | `tm2` | .858 | .586 (0.97) | .842 | .515 (1.13) | .757 | .518 | .497 | .585 |
| Qwen3.5-9B | – | – | – | – | – | – | – | – | – |

PagerDuty/jevlogs/Mind2Web numbers above are this pass's own full-file `eval_wf` extraction and can differ a few
points from a given release card's train-time-pass number for the same checkpoint (release cards sometimes quote
a shorter pass — see each card's footnotes); treat these as the full-row figures.

## 4. DecisionMix v2 (`data_wh` hard curriculum, `data_u` uncertainty corpus) — trained checkpoints only

| model | checkpoint | wh family | wh grammar | wh style | wh level-7 | wh rubric-flip | u ChaosNLI | u real held-out | u synthetic |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | `ts1b` | .818 | .884 | .888 | .493 | .801 | .548 | .649 | .916 |
| Qwen3-4B | `tm1b` | .865 | .880 | .886 | .544 | .833 | .517 | .658 | .907 |
| Qwen3-8B | `ladder_8b` | – | – | – | – | – | – | – | – |
| Qwen3-14B | `tl1b` | .893 | .904 | .898 | **.620** | .847 | .620 | .728 | .907 |
| Qwen3.5-4B | `tm2` | .889 | .886 | .896 | .540 | .835 | .525 | .694 | .909 |

`ladder_8b` predates DecisionMix v2 (6A recipe) and was never trained or evaluated on `data_wh`/`data_u`.

## 5. JevBench (public-subset, 72 standard / 48 easy / 111 hard; see "how to read this")

| model | checkpoint | release status | std acc (Brier) | easy acc (Brier) | hard acc (Brier) |
|---|---|---|---|---|---|
| Qwen3-1.7B | `ts1b` | released | .694 (.40) | 1.00 (.005) | .432 (.79) |
| Qwen3-4B | `tm1b` | released | .806 (.30) | 1.00 (.012) | .423 (.77) |
| Qwen3-8B | `ladder_8b` | trained | .667 (.55) | 1.00 (.015) | .396 (.83) |
| Qwen3-14B | `tl1b` | candidate | **.931** (.18) | 1.00 (.005) | .450 (.66) |
| Qwen3.5-2B (frozen, 3-shot) | `jev_zs3_q35_2b` | frozen control | .583 (.52) | .979 | .387 (.67) |
| Qwen3.5-2B (frozen, semif render) | `jev_zs_q35_2b_semif` | frozen control | .597 (.53) | .958 | .441 (.76) |
| Qwen3.5-4B | `tm2` | trained | .861 (.22) | 1.00 | **.495** (.72) |
| Qwen3.5-4B (frozen, 3-shot) | `jev_zs3_q35_4b` | frozen control | .764 (.36) | 1.00 | .495 (.60) |
| Qwen3.5-4B (frozen, semif render) | `jev_zs_q35_4b_semif` | frozen control | .847 (.24) | .979 | .468 (.62) |
| Qwen3.5-9B (frozen, 3-shot) | `jev_zs3_q35_9b` | frozen control (`tl2` trained, not yet evaluated) | .806 (.29) | 1.00 | .541 (.55) |
| Qwen3.5-9B (frozen, semif render) | `jev_zs_q35_9b_semif` | frozen control | **.931** (.11) | 1.00 | **.595** (.51) |
| Qwen3-14B (frozen, 3-shot, reference) | `jev_zs3_14b` | frozen control | .819 (.28) | 1.00 | .559 (.56) |

The Qwen3.5-9B semif-rendered frozen control ties the trained `tl1b` on standard accuracy and beats every trained
checkpoint in this project on hard accuracy, with zero training — see `PROJECT.md` §1/§5 for the reading.

## 6. Single-decision latency (same-pod, same-torch-build; REPORT §3ab "apples-to-apples")

| model | checkpoint | K=2 (ms) | K=32 (ms) | K=256 (ms) | peak memory K=2→256 (GB) |
|---|---|---|---|---|---|
| Qwen3-1.7B | `ts1b`-shape recipe | 45 | 46 | 106 | 6.4 → 9.3 |
| Qwen3-4B | `tm1b` | 57 | 58 | 96 | 14.6 → 18.6 |
| Qwen3-8B | `ladder_8b` | 70 | 71 | 126 | 29.3 → 34.1 |
| Qwen3-14B | `tl1b` | 59 | 62 | 156 | 51.6 → 57.5 |
| Qwen3.5-4B | `tm2` | – | – | – | – |

`tm2`'s in-process training-repo latency was not benchmarked; its *serving-path* latency (public `inference/`
package, reference DeltaNet kernels) is 34–46 ms warm p50 (REPORT §3ag, `runs/serve_bench2/`) — not directly
comparable to the in-process numbers above, which use a different harness.

## 7. Public releases (deployable today)

| release | HF repo | backbone | JevBench std/hard | inference package | demo |
|---|---|---|---|---|---|
| `typical-small-preview` | `OzLabs/typical-small-preview` | Qwen3-1.7B-Base | .750 / .387 | `inference/` | `demo/app.py` |
| `typical-small` | `OzLabs/typical-small` | Qwen3-1.7B-Base | .694 / .432 | `inference/` | `demo/app.py` |
| `typical-medium` | `OzLabs/typical-medium` | Qwen3-4B-Base | .806 / .423 | `inference/` | `demo/app.py` |
