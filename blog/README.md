# blog/

Two public posts, meant to publish together.

- `typical-launch.md` — **"Typical: Models That Decide, Not Generate."** The launch. Leads with the
  decision primitive, shows the API, the two released models, where they break, and what is open.
  ~1,950 words.
- `technical-deep-dive.md` — **"We Removed Generation from an LLM. Here's What Broke."** The
  archaeology: the candidate-blind architecture that failed, the wrong tap layer, the "none of the
  above" pathology, the long-state data defect, the calibration results, the KV-cache deep copy.
  ~4,400 words. The launch links to it three times.

Every number in both traces to `REPORT.md`, `RESULTS.md`, `COMPARE.md`, or a `releases/*.md` card.
Check the cited section before changing a number, not just the number.

**Read this before you quote any held-out or external number.** Each run directory holds two eval
passes and they are not interchangeable:

- `runs/<run>/results.json` — the **train-time** pass. 1,500 rows, 256-token states. It truncates
  every long external set, so it is wrong for anything with a payload past 256 tokens.
- `runs/<run>/eval_wf_full.json` (or `eval_wf.json`) — the **full-row** pass, every row at
  `--max_state 4096`, `--limit 0`. This is the number to publish.

The gap is not cosmetic. On `ts1b`, PagerDuty reads **.560 train-time and .817 full-row**, and
jevlogs **.500 and .710** — the difference between "below its constant-prediction floor" and "the
first 1.7B model in this project to clear it." Mixing the two has already produced two published
errors: the paper's `tab:app-wf-eval` was built entirely from `results.json` under a caption that
said "full-row pass" (fixed), and the launch post quoted train-time rubric-flip (.803/.832) where
the full-row values are **.718/.743** (fixed). If a number looks surprisingly good on a long
external set, check which file it came from first.

## What the posts may and may not claim

This is the part that goes stale first, so read it before editing.

- **The GitHub repo is private.** The only public artefacts are the Hugging Face model repos
  (`OzLabs/typical-small`, `OzLabs/typical-medium`, `OzLabs/typical-small-preview`), which ship the
  weights, the self-contained `inference/` package, and the eval artefacts. Neither post links
  `REPORT.md`, `RESULTS.md` or `COMPARE.md` as if a reader could open them; the deep dive cites
  report section numbers as provenance and says up front that the report publishes with the
  training code. Do not reintroduce relative links out of `blog/`.
- **There is no `pip install typical`.** PyPI `typical` is an unrelated package (Sean Stewart's
  typing toolkit). The install flow in the launch post is the real one: download the model repo,
  install `inference/requirements.txt`, put `inference/` on `sys.path`. The posts state plainly
  that a packaged install does not exist yet. See the open naming question below.
- **Licensing. The earlier wording here was wrong and the launch post has been corrected.** It is
  not only the undeclared-license `data_u` portions. Two *trained* sources carry explicitly
  non-commercial licenses: **ANLI is CC BY-NC 4.0** (`facebook/anli`, loaded as training data in
  `data.py`) and **SciQ is CC BY-NC 3.0** (11.7k rows of `data_kb`). Both were previously covered
  by "mostly MIT / Apache-2.0 / CC-BY-4.0 for the trained sources", which is false. On top of that,
  `metaeval/ambient` (trained on) and `metaeval/chaos-mnli-ambiguity` (eval-only) declare no
  license at all. The checkpoints and the Qwen3 backbones are Apache-2.0; whether NC training data
  constrains the weights is unsettled and the posts do not resolve it either way. **Still to do:
  triage LogiQA2, MedMCQA, AQuA and the remaining `data_kb` sources the same way, and correct the
  per-source license tables in all three `releases/*.md` cards.** Don't upgrade any of this wording
  without a licensing review.
- **The truncation story is now a positive result (REPORT §3ak-a, 2026-09-22). This section was
  rewritten; do not restore the earlier "null result" wording.** The completed 2x2 scores both
  1.7B arms on the same 605 held-out long states under both renders, with no truncation at eval.
  Matched-condition diagonal: facts-first .942 vs facts-last .830, **+11.2 points, 95% CI
  [+.077, +.147], z = 6.2, p = 5e-10** (policy_permit +.091, action_select +.138). Three separable
  effects: (1) the truncation fix works; (2) render mismatch is separately real and large, and is
  what put the facts-last arm on its majority floor in the single-render comparison; (3) JevBench's
  long_policy subfamily (n = 19) could resolve neither, returning Fisher exact p = 0.232 with the
  hard aggregate going the other way (.378 vs .396). Both posts now tell the sequence, because the
  interim null was an artefact of the metric and the single-render design, not of the mechanism.
- **Watch the 605 vs 330 distinction.** On all 605 held-out long-state items the facts-last arm goes
  .830 (matched) → .640 (mismatched). The .842 → .615 pair, and the .612 majority floor, are the
  K = 2 `policy_permit` subset, n = 330. Earlier drafts of both the deep dive and this README
  attached the subset numbers to "605 items". Do not reintroduce that.
- **Use the paired test on JevBench, never overlapping marginal intervals.** Both posts previously
  wrote "the intervals overlap, so we won't claim the frozen model wins." That is the test the
  paper explicitly warns against (Table `tab:tl1b` caption): marginal intervals over the same 111
  items should not be differenced by eye. The paired cluster bootstrap gives frozen minus `tl1b`
  = **+.108 [+.027, +.189], p = .015** on hard, +.081 [−.009, +.171], p = .082 against
  `tl1b_nokd`, and −.111 [−.250, +.014], p = .10 on standard. State it at that resolution.
- **The frozen 14B leads on the serial families, so "missing data coverage" is not the whole
  story.** Per-family hard, frozen vs `tl1b`: tradeoff .500/.167, long_policy .421/.158,
  ambiguous .714/.429, probability .500/.300, temporal_numeric .333/.200; `tl1b` leads only on
  multi_hop (.611/.444). Earlier drafts said the frozen model "does no better on these specific
  families either", which its own figure data contradicts. Missing coverage explains why training
  did not add the behaviour; it does not explain why training appears to have removed it.
- **Do not say KD contributed nothing.** The matched `--distill_beta 0` control (`tl1b_nokd`) came
  out slightly ahead on hard (.477 vs .450), long-policy (.211 vs .158), standard Brier (.127 vs
  .175) and val NLL (0.410 vs 0.438), but the cluster-bootstrapped hard-tier intervals overlap
  almost entirely (.450 [.360, .541] vs .477 [.387, .568]). The correct claim is that the
  direction is consistent and the effect is undetectable at this sample size. Never re-credit KD
  for the calibration gains either.
- **Sample size.** JevBench public subset: 72 standard items over only 36 independent states (two
  paraphrases per state, so it must be clustered) and 111 hard items, roughly ±9 points on hard.
  Point estimates are fine; comparative claims built on a few points are not. The deep dive
  carries this caveat in its opening and re-states it wherever a claim leans on a small gap.
- **The 14B is not a product.** It stays out of the release table in the launch post and appears
  only as a candidate that missed its own pre-registered bar.
- **Hard-tier probabilities are worse than uniform, and the launch says so.** Under JevBench's
  sum-of-squares Brier, a uniform predictor over the hard tier's candidate sets (K = 2 x38, 3 x15,
  4 x37, 5 x17, 6 x4) scores **.664**. `typical-small` scores .79 and `typical-medium` .77. On the
  standard tier the uniform reference is .689 against our .40 and .30. Any copy that invites
  readers to threshold on confidence has to carry the tier distinction.
- **Level 7 is not "at chance".** The `wh_level7` set is 598 items at K = 2, 224 at 3, 58 at 4, so
  the uniform-guess rate is **.441** (majority-class .250). Scores are .495 / .544 / .620 for
  `ts1b` / `tm1b` / `tl1b`; at n = 880 all three are above the guess rate and the 14B clearly so.
  The finding is the 25-40 point gap to the same models' in-distribution held-out families
  (.84-.90), not the absolute level. Earlier drafts of the paper, both posts and the cards said
  "chance"; that wording is retired.
- **Noul invariance has two true numbers, for two different runs.** The §3ac matched ablation:
  exactly 0 on all **9** sets. The shipped `ts1b` checkpoint: exactly 0 on **10 of 11**, .009 on
  the eleventh, because Noul routes per row and only a candidate set of exactly `{yes, no}` reaches
  the Bernoulli head. Quote the right one for the claim, and don't write "by construction" about a
  whole test suite when the routing is what carries it.

## Where the assets live

Both posts embed figures by relative path (`../figures/fig_*.png`, since they live one directory
below the repo root).

Launch post:

- `fig_architecture` — state encoded once into a KV cache, per-question suffixes read out.
- `fig_hard_families` — JevBench hard-tier accuracy by family: both released models,
  `tl1b`, and the frozen 14B 3-shot control. (Was ladder_14b/tl1b/frozen only, which showed no
  released model in a released-model launch.)
- `fig_latency_quality` — warm per-decision p50 (serve_bench2, K=2, 256-token state, prefix
  cached) vs JevBench standard accuracy, against the latency bands a normal LLM call falls into.
  It previously plotted the *old* `bench_*/bench.json` ladder (a cold single decision on the
  unoptimised serving path, 45-70 ms), which put every Typical point to the right of the
  "one-letter decode" band and argued against the post's own table. Keep the source aligned with
  whatever number the release table quotes.

Deep dive:

- `fig_truncation` — state token length vs truncation cutoffs; long-policy accuracy by checkpoint.
- `fig_calibration` — held-out score NLL and typed-decisions NLL across checkpoints.
- `fig_serving` — cold/warm p50 latency before/after removing the per-decision KV-cache deep copy.
- `fig_ladder` — JevBench standard/hard accuracy across backbone size, trained and frozen.

These are generated by `scripts/make_figures.py` into `figures/<name>.png` (200dpi) and
`figures/<name>.pdf` at the repo root. If a figure is missing when you go to publish, regenerate it
with that script rather than hand-drawing a substitute — the whole point of these posts is that
every number and every chart traces to a run in `runs/`.

## How to publish

Both files render as-is on GitHub (relative image paths, standard Markdown tables) and convert
cleanly to the usual targets:

1. **Hugging Face blog (`hf.co/blog`) or any static-site generator (Jekyll, Hugo, Ghost,
   Substack).** Copy each post's body (everything after the closing `---`), re-export the front
   matter (`title`, `date`) into the target's schema, upload the `figures/fig_*.png` files (not the
   `.pdf` versions) to the target's media library, and swap the `../figures/` paths for the
   resulting URLs. Keep the alt text and captions; they were written to stand alone if an image
   fails to load.
2. **Fix the cross-links.** The two posts link to each other by relative path
   (`./technical-deep-dive.md`, `./typical-launch.md`). Point those at the published URLs.
3. **Company site / marketing CMS.** Same as above. These assume a technical reader and link
   straight to HF model cards. Don't trim those links to make it more "marketing"; the promise of
   both posts is that every claim is checkable.

## Open question for a human

Our inference package is imported as `from typical import Typical`, and PyPI `typical` v2.9.0 is
an established, unrelated package. A future `pip` release needs either a different distribution
name (with the import name possibly following) or a conversation with that project's maintainer.
Both posts avoid the collision by shipping the package inside the model repos, so nothing is
blocked, but it needs deciding before any packaged install ships.

## What goes stale first

In rough order of how fast it rots:

1. **The live Hugging Face model cards.** They are the first thing a reader opens and the thing
   most likely to disagree with the posts, because they are edited on the hub and in
   `releases/*.md` independently. Before publishing, diff the live card against `releases/*.md`
   against the post, on: the external floors (PagerDuty / jevlogs), the latency figure and what
   protocol it was measured under, anything that says "calibrated", the level-7 wording, and the
   license table. Latency in particular has two legitimate numbers that are not the same quantity:
   the `§3ab` apples-to-apples ladder (45 ms single decision, K = 2, 256-token state, state encode
   included, unoptimised path) and the serving number the posts quote (15.5-17 ms warm p50, prefix
   already cached, `runs/serve_bench2`). Say which one, every time.
2. **The JevBench leaderboard and version.** The posts cite v1.2.1 and a 231-id public subset. The
   board gains entries and revises item counts; re-check before publishing and again before
   linking anyone to a placement.
3. **Anything about what is public.** Training code, the report, a `pip` install.

## Updating them later

If a number changes (a new release, a fixed bug, a benchmark rerun), edit the post directly and
re-check the specific `REPORT.md`/`RESULTS.md`/`COMPARE.md` section cited next to the claim, rather
than just bumping the number. One edit is already queued: the launch post's 14B paragraph and the release
table, if `typical-large` ever clears its pass rule.
