# Interactive explainer plan (research-communication lead, 2026-09-22)

Ground truth checked on disk: `site/data/latency.json` (`native_sweep[L][K]`, L in {256, 1000, 2000}, K in {2..256}, M in {1, 32, 256}; L=2000 stops at K=32), `site/data/reliability.json`, `site/data/presets.json` + `site/data/replays.json` (real typical-small outputs, MPS), 134 `runs/*/results.json` (75 distinct eval sets; floors live only in `runs/ts1b/eval_wf_full.json` / `runs/tm1b/eval_wf.json` `baselines.majority_acc`), `inference/typical/native.py` (the forward path), `site/js/charts.js` (560 lines of hand-rolled SVG, dot system, no libraries).

## 1. The five pieces, ranked by insight per effort

### P1. Latency explorer (native L x K x M grid)
- Visitor: three segmented steppers (state length 256/1000/2000, K 2..256, M 1/32/256). Each click re-lights two dot rows: native vs batched log-prob prompting, plus a "ms per extra question" tile. Hover a bar for `t_state_ms` + `t_queries_ms`.
- Learns in 10 s: one decision costs ~45 ms almost regardless of K; extra questions on a cached state cost ~3-4 ms each; prompting costs ~11 ms each and grows with K.
- Data: `site/data/latency.json` -> `native_sweep[L][K].native[i].{m,t_state_ms,t_queries_ms,total_ms,marginal_ms}`, `.baseline[i]`, `.peak_mb`, `.crossover`; release ladder from `latency.json.ladder.<model>.single_ms/marginal_ms_m32`.
- Effort: 1 day. All data on disk.

```
 state tokens [256][1000][2000]   candidates K [2][4][10][32][64][128][256]   questions M [1][32][256]
 native         ●●●●●●●●●●●●●○○○○○○○   45.3 ms   (state 19.7 + queries 45.3, one decision)
 prompting      ●●●●●●●●●●●●●●●●○○○○   56.0 ms
 per extra question, state cached: native 4.0 ms · prompting 11.0 ms     peak 11.2 GB
 bench: runs/bench_native_L256c8 (nc_n3 research ckpt, Qwen3-1.7B-Base, one H100)
```

### P2. Calibration explorer with the ECE arithmetic
- Visitor: toggles model (small / medium / preview) and tier (standard / hard). Hover a bin: `n/N x |acc - conf| = contribution`; a running sum under the chart lands on the ECE number.
- Learns: standard tier is honestly calibrated (.11 / .09); the hard tier is over-confident (.26 / .28) and at chance, and the arithmetic is nothing more than a weighted gap.
- Data: `site/data/reliability.json[model][tier][]` (`lo,hi,n,mean_confidence,accuracy`), `models.json[].jevbench[tier].ece`, `chance.json[tier].acc`.
- Effort: half a day; extends `charts.js:reliability()`.

```
 [typical-small][typical-medium][preview]   [standard][hard]        n = 72
 acc │            ▒▒     bin .9-1.0: n 33/72 x |.879 - .964| = .039
     │       ▒▒  ▒▒     bin .8-.9 :  n 10/72 x |.700 - .845| = .020
     │  ▒ ▒▒ ▒▒  ▒▒     ...
     └──────────── conf  sum = ECE .114   chance .311 ──
```

### P3. Architecture trace ("trace one decision")
- Visitor: scrubs a rail through the real forward pass on the playground ticket (#7734): state tokens -> `[eos]+state` into the truncated trunk (20 of 28 layers, LoRA on 13-20) -> KV cache (Ls tokens) -> one suffix per question (rendered exactly as `_render_letters_nonull` or `_render_query_only`) -> `h_D` at the trailing eos + pooled candidate spans -> head (bilinear score + factored null gate, or Bernoulli sigmoid for Noul) -> probabilities + p_null. Toggle among the four questions; the Noul question shows the label-order swap giving identical P(yes).
- Learns: the state is read once; every question is a short suffix; nothing is generated; the abstain is a gate, not a candidate.
- Data: new `site/data/trace.json` (below) built from `presets.json.playground` and `replays.json["1wzx452"]` (probs, p_null, `ms` 211.8, `device` mps); stage timings for the H100 annotation from `latency.json.native_sweep["256"]["4"].native[0]`.
- Effort: 2-3 days (script half a day, SVG graph 2 days).

```
 ○────●────○────○────○────○   scrub    question [1 team][2 urgency][3 escalate][4 want]
 state   trunk    cache   suffix   head    probs
 "Ticket #7734: A customer writes: 'I was charged twice..."   36 tokens (+1 eos sink)
 ┌ trunk: Qwen3-1.7B, layers 1-20 of 28, LoRA r16 on 13-20 ┐  19.7 ms (H100, one-off)
 └ KV cache: 37 positions, reused by every question         ┘
 suffix 2: "How urgent is this ticket, 0=low..3=critical?\nA. 0\nB. 1\nC. 2\nD. 3\n" + eos   35 tokens, T=36
 head: s_k = u·v_k ; r = σ(gate(top2, margin, mean, lse, h, mean_c, var_c)) ; P(k)=(1-r)p_k, P(∅)=r
 0 ●○○○○○○○○○ .018   1 ●○○○○○○○○○ .110   2 ●●○○○○○○○○ .226   3 ●●●●●●○○○○ .647   ∅ .001   E[idx] 2.50
 replayed from replays.json (typical-small, MPS, 211.8 ms for all four)
```

### P4. Run explorer (with the timeline folded in as its x-axis)
- Visitor: pivot A "pick an eval set, see every run" (dots ordered by run date, floor/chance rule drawn, released runs in champagne, probes/ablations hollow); pivot B "pick a run, see every set" (row of 75 dots with hover). The date axis carries the nine bug markers and the eight verdict markers from `research-timeline.json`, so the "what changed" strip is this chart's ruler, not a separate piece.
- Learns: what actually moved SNLI (tap 20), CLINC (factored null), PagerDuty (typed heads), and that most of the 134 runs are controls that did not move anything.
- Data: new `site/data/runs-index.json` (below); `research-timeline.json.bugs/verdicts`.
- Effort: 3 days (index script 1, chart 2).

```
 set [snli_test ▾]  metric [acc][nll]  show [lineage][ablations][probes][baselines]
 1.0 │                      ●  ●    ●●  ◆      ◆ = released
     │  ○  ○   ●●  ●  ○ ●
 .58 ├──hyp-only floor─────────────────────────   hover: nc_v3_tap20 .906 · tap 20 · 12k steps · §3t
 .33 ├──chance 1/3─────────────────────────────
     └ 09-16 ─ ▲bug4 ─ 09-18 ─ ▲verdict 3j ─ 09-20 ─ 09-21
 run [ts1b ▾] → 75 dots, one per set, floor tick under each
```

### P5. Training-curve explorer
- Visitor: picks runs by size (1.7B / 4B / 8B / 14B) and recipe tag (energy / native / ladder / release), sees `train/loss` and `val/nll` aligned by step; drag on x to zoom; pinned comparison ts1 (collapsed Noul routing) vs ts1b, r1_bs16 vs r1_cand, ladder_4b/8b/14b.
- Learns: the released models were 12k steps at bs 64; the bug runs are visible as curves, not just as prose; small-batch under-fitting is a shape you can see.
- Data: `site/data/curves/<run>.json` from W&B (schema below). Not on disk yet, hence last.
- Effort: 2-3 days plus the export.

```
 size [1.7B][4B][8B][14B]   recipe [energy][native][ladder][release]   pinned: ts1 vs ts1b
 loss │╲
      │ ╲__            ts1b ───   ts1 ┈┈┈   r1_bs16 ···
      │    ╲_______________________
      └───── step 0 ─── 4k ─── 8k ─── 12k   [drag to zoom]   val/nll at 1k, 2k, ...
 source: W&B pcdm/<run_id>, exported 2026-09-xx; runs without a curve: results.json only
```

### Cut
- (e) standalone timeline strip: already exists as `charts.js:timeline()`; its content becomes P4's x-axis. A second scrubbable copy would be the same eight points twice.
- (f) data-mix explorer: the donut and row bars exist; example rows per family need `data_wh/*.jsonl` from the training branch, which is not in this worktree, and publishing training rows next to held-out claims invites a leak argument. Keep the static mix chart.

## 2. Interaction spec, states, honesty, mobile

Common: every piece has a `source` line under it (file + key path, rendered from the JSON's own `source` field, never typed by hand in HTML), a loading state (dot-matrix skeleton, caption "loading data/x.json"), and an error state (caption "data/x.json missing; chart withheld" instead of an empty box). Unreleased runs are always hollow markers with "(not released)" in the label, which `accentFor()` already turns cream instead of champagne. All numbers three decimals as in source. Hover on desktop is tap on mobile; nothing depends on hover alone.

P1: steppers are discrete (only measured cells; no interpolation, no continuous slider). Cells absent from the bench (L=2000, K>32) render as "not measured (OOM in baseline)". Caption states the bench checkpoint is `nc_n3` (same shape as the release, not the release) and that the release ladder numbers (45/56/60 ms) come from `latency.json.ladder`. Never live: the local MPS server has its own ms, which stays in the demos. Mobile: steppers stack vertically, bars keep 24 dots.

P2: model x tier toggles; bins with n = 0 render dimmed with "n 0". Running sum visible at all times, ending in the published ECE; if the recomputed sum differs from `models.json` ECE by > .001 the chart shows both and flags it. D1 disclosure within one viewport. Mobile: chart 100% width, arithmetic list under it.

P3: scrub is a single pointer handler over six stops; keyboard arrows move it. Static by default (replays.json); if `api.js` finds the local server, a "live" badge appears and the probabilities re-run, with device + ms shown. Token counts come from the real tokenizer at precompute time, not estimated. The head stage prints the formulas exactly as in `native.py:factored_null_logits` and `_bern_probs`. Mobile: the six stops become a vertical list; the scrub becomes "next / previous".

P4: two pivots, one selected at a time. Runs with `eval_only: true` or `eval_cap` < full or `max_state` != 1024 show a protocol badge on hover ("256-token states"), because those numbers are not comparable to the release rows. Floors only where a source exists: `majority_acc` from eval_wf files for wf/wh/external sets, `chance.json` for JevBench, 1/K for fixed-K sets (K listed in the script); sets without a sourced floor get no floor line and say so. Probe/baseline/smoke runs default off. Mobile: pivot A only, dots become a sorted list.

P5: runs without an exported curve are listed with "no curve, results only". Curves show the run's own `steps`/`bs`/`grad_accum` next to it, so bs 16 vs 64 is not read as the same x. No smoothing toggle: the exported points are the logged points, downsampled by stride. Mobile: one series at a time, size buttons only.

## 3. Data pipeline

`scripts/export_wandb.py` (plan; do not run now): `wandb.Api().runs("guy-na8/pcdm")`, for each run write `site/data/curves/<run.name>.json`:

```json
{"run": "ts1b", "wandb_id": "abc123", "exported": "2026-09-22", "state": "finished",
 "args": {"backbone": "...", "steps": 12000, "bs": 64, "grad_accum": 1, "lr": 3e-4, "lora_lr": 1e-4,
          "tap_layer": 20, "family_weights": "...", "max_state": 1024, "seed": 0},
 "tags": {"params": "1.7B", "recipe": "release"},
 "series": {"train/loss": [[step, v], ...], "train/lr_lora": [...], "train/step_time": [...],
            "train/tokens_per_s": [...], "val/nll": [...], "val/...": [...]},
 "source": "wandb://guy-na8/pcdm/<id>"}
```
Stride-downsample to <= 400 points per series, round to 4 significant figures; `site/data/curves/index.json` lists run, tags, and which series exist. Expected ~15 KB per run, ~1 MB total, fetched lazily per selection. `tags.recipe` derives from `args.readout` / `args.nc_head` / run-name prefix (`ladder_`, `ts`/`tm`), same rule as the runs index so the two pieces agree.

`scripts/precompute_research.py` extensions:
- `runs-index.json`: for every `runs/<d>/results.json` (skip `smoke*`, `bench_*`, `jev_*`): `{name, date (git log --diff-filter=A on gpu-runpod-full-experiment), group (lineage|ablation|probe|baseline|zero_shot by prefix), released_as, args subset (as above + nc_render, null, noul_head, score_head, eval_only, eval_cap, readout), eval: {set: {acc, nll, ece, null_recall, n}}}` from `eval.<set>.raw`; `floors: {set: {value, source}}`; `sets: [...]` ordered by the eval-suite families in research.md. Estimate 134 x 75 x 5 numbers at 4 sig figs, about 350 KB; hard-fail the script if > 1 MB.
- `trace.json`: tokenize `presets.playground.state` with the Qwen3-1.7B tokenizer, render each query with `RENDERS[...]` imported from `inference/typical/native.py` (bern rows via `_is_bern_row`), store tokens, spans, `Ls`, `T`, `Kmax`, and copy probs/p_null/ms/device from `replays.json` under the same hash `record_replays.py` computes. Needs only the tokenizer, no weights.

Traceability: each JSON carries `source` strings pointing at file + key path; `research.js:sourceNote()` already renders them; the build refuses a chart whose data lacks `source`.

## 4. Tech

Hand-rolled SVG, no chart library, no GSAP. `charts.js` already has scales, ticks, the dot vocabulary, `reliability()`, `lineChart()`, `timeline()`; the new interactions are one pointer handler each (scrub, x-brush, hover) and segmented buttons, which is under 100 lines per piece. D3 would add ~280 KB for scales we have; uPlot and Plot are canvas or their own CSS and fight the dot system and the "no load animation" rule. Choreography is a CSS transition on transform plus `dots.js:eased()`, which already animates dot fills. Budget: total new JS under 40 KB unminified, zero dependencies, `package.json` stays build-free.

## 5. Copy scaffolding

P1. Looking at: measured milliseconds on one H100 for one decision and for each extra question on the same cached state, native head versus prompting the same backbone. Takeaway: one decision costs about 45 ms whatever K is, and each extra question costs a few milliseconds because the state is read once.

P2. Looking at: the accuracy of each confidence bin on the JevBench public subset, with the weighted gap that makes up the ECE. Takeaway: calibration holds on the standard tier and breaks on the hard tier, where both models are also at chance.

P3. Looking at: the real forward pass on one support ticket, from tokens to probabilities, replayed from a recorded run of typical-small. Takeaway: the state is encoded once, each question is a short suffix, and abstain is a gate in the head rather than a candidate in the list.

P4. Looking at: every evaluation set for every run of this project, ordered by date, with the constant-prediction floor where one exists. Takeaway: a handful of fixes moved the numbers and most runs were controls that did not.

P5. Looking at: training loss and validation NLL against step for the runs logged to W&B, selectable by size and recipe. Takeaway: the released models are twelve thousand steps at batch 64, and the failed runs are visible as curves rather than as footnotes.
