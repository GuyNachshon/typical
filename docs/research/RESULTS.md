# PCDM / Typical — results (2026-09-22)

Regenerated as one set of tables per model-family size, Qwen3 and Qwen3.5 side by side. Every number is pulled
from `runs/*/results.json` / `runs/probe_*/results.json` / `runs/jev_native_*/summary.json` /
`runs/bench_*/bench.json` (local, or `guychuk/pcdm-runs` where not pulled locally) — none are copied from prose
summaries or estimated. Missing cells are `–`, not a guess. Narrative and verdicts: `REPORT.md`; plan: `docs/plan/PLAN7.md`;
index: `docs/plan/PROJECT.md`; release cards with full training args: `releases/*.md`.

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

## 3. Held-out workflow (W), typed-decisions, and external floors — capped `results.json` pass

| model | checkpoint | held-out noul | held-out score (NLL) | held-out style | typed-decisions acc (NLL) | PagerDuty (floor .792) | jevlogs (.697) | Mind2Web (.427) | tree-choice |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | `ts1b` | .710 | .522 (1.01) | .861 | .479 (1.27) | .560 | .500 | .351 | .469 |
| Qwen3-4B | `tm1b` | .811 | .528 (1.03) | .861 | .540 (1.17) | .553 | .480 | .547 | .683 |
| Qwen3-8B | `ladder_8b` | .681 | .543 (1.67) | .882 | .407 (2.14) | .321 | .501 | .404 | .411 |
| Qwen3-14B | `tl1b` | .831 | .621 (0.95) | .882 | .600 (1.04) | .805 | .514 | .564 | .506 |
| Qwen3.5-2B | – | – | – | – | – | – | – | – | – |
| Qwen3.5-4B | `tm2` | .858 | .586 (0.97) | .842 | .515 (1.13) | .757 | .518 | .497 | .585 |
| Qwen3.5-9B | – | – | – | – | – | – | – | – | – |

**Protocol note (corrected 2026-09-23).** Every number in this table is the capped `results.json`
evaluation pass, not a full-file `eval_wf` run — the heading previously said otherwise. The two protocols can
disagree by far more than a few points on the external suites: `ts1b` reads PagerDuty **.560** here, while
full-file `eval_wf` passes over the same suite (n = 6,000) land between .69 and .79 on comparable checkpoints.
Compare rows within this table to each other, and do not compare them against a full-file number from §5 or
from REPORT.md's §3x re-evaluation.
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

## 5a. JevBench with confidence intervals — what this benchmark can and cannot resolve (2026-09-22)

Cluster-bootstrapped over the paraphrase `group` (`scripts/jev_ci.py`): the standard tier is 72 items but only
**36 independent states**, each appearing as two paraphrases, so item-level resampling would understate the
interval. Hard has one item per group. 20,000 resamples.

| run | standard | 95% CI | hard | 95% CI |
|---|---:|---|---:|---|
| `ts1b` (typical-small) | .694 | [.569, .819] | .432 | [.342, .523] |
| `tm1b` (typical-medium) | .806 | [.694, .903] | .423 | [.333, .514] |
| `tm2` (Qwen3.5-4B) | .861 | [.778, .944] | .495 | [.405, .595] |
| `tl2` (Qwen3.5-9B) | .833 | [.722, .931] | .495 | [.405, .586] |
| `ladder_14b` | .875 | [.792, .944] | .468 | [.378, .559] |
| `tl1b` (14B, KD) | .931 | [.861, .986] | .450 | [.360, .541] |
| `tl1b_nokd` (14B, no KD) | .917 | [.833, .986] | .477 | [.387, .568] |
| `ts1b_semif` (1.7B, SemIf render) | .792 | [.667, .903] | .441 | [.351, .532] |
| `trunc_first` (ablation arm A) | .750 | [.625, .861] | .378 | [.288, .468] |
| `trunc_last` (ablation arm B) | .736 | [.597, .861] | .396 | [.306, .486] |

**Read this table before reading any other JevBench comparison in this repo.** Hard is ±9 points at n = 111 and
standard is ±6–13 points at n_eff = 36, so *every* adjacent pair above is statistically indistinguishable.
Paired per-item tests are more powerful than differencing these intervals and are the only JevBench comparisons
worth quoting:

| paired comparison | diff | 95% CI | p |
|---|---:|---|---|
| frozen 14B (3-shot) − `tl1b`, hard | +.108 | [+.027, +.189] | **.015** |
| frozen 14B (3-shot) − `tl1b_nokd`, hard | +.081 | [−.009, +.171] | .082 |
| `tl1b_nokd` − `tl1b` (KD off vs on), hard | +.027 | [−.027, +.090] | .45 |
| `ts1b_semif` − `ts1b`, standard | +.097 | [−.056, +.250] | .23 |
| `tl2` (9B) − `tm2` (4B), hard | .000 | [−.090, +.090] | 1.00 |

`tl2` vs `tm2` are both exactly 55/111 but are *not* the same model: zero of 111 probability vectors match and
they disagree on 26 items (13 each way). The benchmark cannot resolve 4B vs 9B here; that is not a tie.

## 5b. Long-state performance — the axis JevBench cannot see (2026-09-22)

605 held-out long states (`scripts/make_long_eval.py`), identical items in both columns, differing only in where
the `Case:` block sits. No truncation at eval (window 4,096; state p50 1,965 tokens). Majority-class floors:
.334 overall, .612 on the `policy_permit` (K=2) family.

| checkpoint | recipe | facts-**first** | facts-**last** | policy_permit (first) |
|---|---|---:|---:|---:|
| `typical-small` (**released**) | pre-fix | **.598** | .798 | .536 |
| `ts1c` (1.7B) | post-fix | **.947** | .790 | .948 |
| `typical-medium` (**released**) | pre-fix | **.612** | .866 | .555 |
| `tm2` (Qwen3.5-4B) | post-fix | **.950** | .879 | .967 |
| `ladder_14b` | pre-fix | .851 | .919 | .839 |
| `tl1b_nokd` (14B) | post-fix | **.997** | .921 | .997 |

**Both public checkpoints carry a 20–25 point deployment trap, and both already have a fixed replacement.**
Callers write their own state text; if the case facts go *before* the policy body — the natural ordering — the
released models lose 20–25 points and land at or near the majority-class floor on the yes/no family. The post-fix
checkpoints do not: `ts1c` is +34.9 over `typical-small` and `tm2` is +33.8 over `typical-medium` on facts-first,
while giving up nothing on facts-last (−0.8 and +1.3 respectively). They are strictly better or equal.

**None of this is visible on JevBench.** `ts1c` reads .708/.432 there against `ts1b`'s .694/.432 — statistically
indistinguishable — while being 35 points better on the axis the fix targeted. `ladder_14b` reads .053 on the
JevBench `long_policy` family and .919 here in its own matched render. Treat JevBench as one suite among several,
not as the release gate; see §5a for why its intervals cannot support the comparisons it was being used for.

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
checkpoint in this project on hard accuracy, with zero training — see `docs/plan/PROJECT.md` §1/§5 for the reading.

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
| `typical-small` **v2** (`ts1c`) | `OzLabs/typical-small` | Qwen3-1.7B-Base | .708 / .432 | `inference/` | `demo/app.py` |
| `typical-medium` **v2** (`tm2`) | `OzLabs/typical-medium` | **Qwen3.5-4B-Base** | .861 / .495 | `inference/` | `demo/app.py` |

**Superseded 2026-09-23 — both public checkpoints now carry the fixed-corpus weights (§5b).** They lose 20–25
points on long states when the caller puts case facts before the policy body. Drop-in replacements already exist
and need no further training — only a card and an upload:

| supersedes | replacement checkpoint | long-state facts-first | gain |
|---|---|---:|---:|
| `typical-small` | `ts1c` (1.7B, post-fix recipe) | .947 vs .598 | **+34.9** |
| `typical-medium` | `tm2` (Qwen3.5-4B, post-fix recipe) | .950 vs .612 | **+33.8** |

Until those ship, the model cards must state the ordering caveat: on long documents, put the case facts **after**
the policy body. (Added to all three cards, README and the launch post on 2026-09-23.)

**Release prerequisite, checked 2026-09-23.** The two candidates are not symmetric:

| candidate | backbone | ships with the current public inference package? |
|---|---|---|
| `ts1c` → `typical-small` v2 | Qwen3-1.7B-Base (unchanged) | **yes**, no code change needed |
| `tm2` → `typical-medium` v2 | **Qwen3.5**-4B-Base (generation change) | **no** — needs `inference/typical/backbone.py` refreshed first |

The published `inference/` package is current except for `backbone.py`, which predates the Qwen3.5 port: it lacks
the VL-wrapper unwrap, the Gated-DeltaNet LoRA target names and the `linear_attn` parent walk, so a Qwen3.5
checkpoint fails to load. `pcdm/native.py`, `core.py`, `__init__.py`, `example.py` and `requirements.txt` are byte-identical
to local, so the serving optimisations are already public and the shipped Qwen3 models are not running stale code.
The `backbone.py` change is purely additive — every new branch is `hasattr`-guarded, so no Qwen3 path changes — but
it must be published alongside (or before) any Qwen3.5-backed release or users get a load failure on first call.
Both backbones are Apache-2.0; there is no licensing blocker either way.
