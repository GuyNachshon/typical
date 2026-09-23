# blog/

Two public posts, meant to publish together.

- `typical-launch.md` — **"Typical: Models That Decide, Not Generate."** The launch. Leads with the
  decision primitive, shows the API, the two released models, where they break, and what is open.
  ~1,950 words.
- `technical-deep-dive.md` — **"How Typical Works, and What Broke Building It."** The system
  piece: the interface, the architecture in depth (tap depth, candidate-aware readout, factored
  null, typed heads, cached prefix), the training recipe (E/K/W/U mixture, counterfactual rubric
  groups, null augmentation, ordinal smoothing, calibration-based checkpoint selection), results
  with confidence intervals and paired tests, the truncation 2x2, rendering as a factor, the
  metric lesson, the serving bugs, releases and open work. Each architecture choice is told
  together with the negative result that forced it. ~5,700 words. The launch links to it twice.

Every number in both traces to `REPORT.md`, `RESULTS.md`, `COMPARE.md`, or a `releases/*.md` card.
Check the cited section before changing a number, not just the number.

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
- **Licensing.** Both released checkpoints are Apache-2.0 over Apache-2.0 Qwen3 base models, and
  the posts say so. The headline claim is "open weights and inference code today, recipe
  documented, training code coming", because two portions of `data_u`
  (`metaeval/ambient`, `metaeval/chaos-mnli-ambiguity`) declare no license on their HF cards. The
  launch post flags that rather than asserting commercial-use safety. Don't upgrade that wording
  without a licensing review.
- **The truncation story is now established, and the way it got established is the point**
  (REPORT §3ak-a/-c/-e, superseding the earlier null reading). Sequence the deep dive tells: a
  verified data defect (98.8% of long rows lost their facts at a 1,024-token window; truncation
  keeps the start and drops the end); then a matched render-order ablation that was underpowered
  on the pre-registered metric (long_policy .316 = 6/19 vs .105 = 2/19, Fisher p = 0.232, CI
  [−0.053, +0.474], hard aggregate going the other way); then a purpose-built 605-item eval and a
  2x2 that separates truncation from train/test render mismatch. Matched-condition gap
  **+.112 [+.077, +.147], p = 5e-10 at 1.7B**, **+.078 [+.055, +.100], p = 7e-12 at 14B**, and
  **+.111 at a second seed**. Render mismatch is a *separate* real effect (23 points, .842 → .615
  onto the .612 floor) and does not explain the truncation effect away. Do not reintroduce the
  "null result" framing.
- **Do not say KD contributed nothing.** The matched `--distill_beta 0` control (`tl1b_nokd`) came
  out slightly ahead on hard (.477 vs .450), long-policy (.211 vs .158), standard Brier (.127 vs
  .175) and val NLL (0.410 vs 0.438), but the cluster-bootstrapped hard-tier intervals overlap
  almost entirely (.450 [.360, .541] vs .477 [.387, .568]). The correct claim is that the
  direction is consistent and the effect is undetectable at this sample size. Never re-credit KD
  for the calibration gains either.
- **Sample size.** JevBench public subset: 72 standard items over only 36 independent states (two
  paraphrases per state, so it must be clustered) and 111 hard items, roughly ±9 points on hard
  and ±6–13 on standard. Every adjacent pair of our own checkpoints is statistically
  indistinguishable there (RESULTS §5a). Only three paired per-item results in the whole record
  survive: frozen-14B-beats-trained on hard (+.108 [+.027, +.189], p = .015), the rendering effect
  (p < .001) and the long-state result. The deep dive carries this in its opening, gives the full
  interval table in §4, and re-states it wherever a claim leans on a small gap.
- **The `long_policy` family is n = 19 and contradicts itself across seeds** (6/19 vs 2/19 at seed
  0, 5/19 vs 5/19 at seed 1). Never quote it as a verdict on anything. Deep dive §7 is about
  exactly this.
- **The 14B is not a product.** It stays out of the release table in the launch post and appears
  only as a candidate that missed its own pre-registered bar.

## Where the assets live

Both posts embed figures by relative path (`../figures/fig_*.png`, since they live one directory
below the repo root).

Launch post:

- `fig_architecture` — state encoded once into a KV cache, per-question suffixes read out.
- `fig_hard_families` — JevBench hard-tier accuracy by family, trained vs frozen 14B.
- `fig_latency_quality` — the family's latency vs JevBench standard accuracy, against the latency
  bands a normal LLM call falls into.

Deep dive:

- `fig_architecture` — also used by the launch post; reused here to open the architecture section.
- `fig_deltaq` — raw among-K accuracy vs question-dependence for every readout probed; the
  candidate-blind negative in one chart.
- `fig_truncation` — state token length vs truncation cutoffs; long-policy accuracy by checkpoint.
  Its right-hand panel is the n=19 JevBench family, so the caption says outright that this is the
  measurement the post argues against steering by.
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

## Updating them later

If a number changes (a new release, a fixed bug, a benchmark rerun), edit the post directly and
re-check the specific `REPORT.md`/`RESULTS.md`/`COMPARE.md` section cited next to the claim, rather
than just bumping the number.

Queued edits:

1. **The launch post is stale on the v2 releases.** It still quotes `typical-small` .694/.798 and
   `typical-medium` .806/.612, and its "one thing to know before you call it" section says the
   fixed checkpoints "will supersede these". `ts1c` and `tm2` shipped as v2 on 2026-09-23
   (REPORT §6.5, RESULTS §7), and the deep dive says so, so the two posts currently disagree.
   Update the launch post's release table, the facts-first/facts-last table and that paragraph
   before publishing the pair.
2. The launch post's 14B paragraph and release table, if `typical-large` ever clears a pass rule —
   and note that the rule itself is being restated on the n=605 set and paired tests (REPORT §6.3).
