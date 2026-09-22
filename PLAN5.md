# Typical site v10 — the brief

Synthesised from `.context/strategy/{cmo,cpo,design,explainer}.md` (2026-09-22). Reference lane:
`designs/BROWSERBASE_DESIGN.md` (broadsheet meets data terminal), every noun swapped. Supersedes the
style section of PRODUCT.md and everything in DESIGN.md.

## 1. Positioning (CMO)
- Category noun: **decision model** (the market already uses it). We own the modifier *open* and the
  posture *every number with its floor*. Never name Jev on the index; differentiate on what a hosted
  API cannot be: open weights, runs locally, runtime-defined candidates, an explicit ∅, printed controls.
- Headline: **"Small open models that [decide]."** (highlight box on *decide*). Sub: "Text state and a
  typed question in. One probability per candidate plus an explicit ∅ out, in one forward pass: 45 ms on
  one H100, a few milliseconds per extra question on the same state. 1.7B and 4B weights on Hugging Face."
- Hero proof points (ledger-safe, floors printed): 45 ms / 2.7 ms per extra question · CLINC-150 .804 /
  .847 with OOS abstain recall .827 · held-out rule families .836 / .874, rubric flip .718 / .743.
  JevBench stays in Results next to its disclosure.
- Objections pre-empted in our words: hard tier at chance and over-confident; text-only (games read
  pre-computed sentences); not an LLM (no arithmetic, ≤ 1k tokens); Choice is order-robust, not
  invariant; two sizes, three checkpoints; local ms ≠ H100 ms.
- Voice: plain, exact, a little dry. No em dashes, no triads, no verbs the numbers don't back.
  Buttons say "Run" (live) / "Replay" (recorded); the word "static" never appears.

## 2. The idea (lead designer): ink fills to p
Every decision on the site is drawn the same way: a rectangle whose fill covers exactly p. The
headline highlight box is that box at p = 1; in the hero it fills to the live winning probability; a
stat number sits in a box filled to its own value (`.804` → 80.4%); a decision card's candidate labels
are the bars; ∅ is the dashed empty box. At scale the same device is **the tape**: a running canvas
band of the 1,112 recorded decisions in `site/data/replays.json`, one column per decision, argmax cell
in yellow, ∅ dashed. Nothing yellow on the page is decorative; every filled pixel is a value.

## 3. Identity
- Palette (restrained, one signal): canvas `#FAF8F1`, ink `#17150F`, grey-1 `#5E5A50`, grey-2
  `#D8D3C6`; pastels straw `#F4E8BF` (decision cards), sage `#DDE7D3` (code / model windows), fog
  `#E5E6E0` (table stripes), clay `#EFDCCB` (limits); **signal = Board Yellow `#F2C400`**
  (`oklch(0.837 0.171 92)`). Yellow only inside fill-to-p boxes, the tape's argmax cells and the footer
  band; never text, button, border, icon, hover or gradient. Chosen against Browserbase orange (h 35),
  typesafe salmon/magenta, Hume violet, Together periwinkle: h 92 is unclaimed and keeps ink legible at
  full chroma, which the device needs.
- Type: display **Technor** 400/500 (Fontshare), body **Host Grotesk** 400/500, mono **Martian Mono**
  wdth 87.5 (Google). Scale: caption 12 mono caps · body 16 · subheading 22 · heading 34 · heading-lg
  56 · display clamp(64px, 9vw, 136px), tracking −0.02 → −0.04em as size grows. ∅ is a CSS-drawn mark
  (`.nul`), also the favicon and the wordmark's companion.
- Surfaces: no shadows anywhere; hairlines 1px grey-2 on canvas, 1px ink on pastel; radii cards 4,
  tags 999, buttons 50; 4px spacing base (4 8 12 16 24 32 48 64 96 160 224); page 1200, band 1440.

## 4. Information architecture (CPO), one page + research
0 Nav: ∅-mark + Typical · Results · Demos · Research · Code · ghost "Hugging Face →" · primary
  "Try it live" · mode word `recorded` / `live · MPS`.
1 Hero: caption "SMALL OPEN DECISION MODELS · 1.7B / 4B · OPEN WEIGHTS"; headline with the live box;
  sub; two pills; **proof strip** (three stat tiles, number in a fill-to-p box, floor + source under
  each); right column: the decision card cycling real recorded decisions (live ones when a server is
  up). Below, full-bleed: **the tape**, running. At 1440×900 everything the 10-second rule needs is in
  this viewport; at 390 the strip is one row of three numbers over the card.
2 Results: register (JevBench with the frozen-backbone column; evidence/intent), D1 verbatim directly
  under it, the scaling ladder and latency-vs-K, calibration (standard tier; hard tier on research),
  one "Sources" line.
3 How it works: the three primitives as **model windows** (sage cards with a miniature decision card
  each), six lines of Python, the "what it is not" paragraph, the **∅ toggle** (gold absent / present,
  recorded pair) as the comprehension device, and a compact **architecture trace** (P3) that scrubs a
  real ticket through trunk → cache → suffix → head → probabilities.
4 Demos: DOOM (real, wide, primary) → Snake + Rules-flip row → Try it (editable card; presets when
  recorded) → secondary rows that open inline: Drive, SQL, Doc-20. Every demo renders the one decision
  card. Ads and Compaction move to research; Compaction keeps one line in "What fails".
5 What fails: five one-line items with numbers (aiming from bearing sentences, "or"/negation rules,
  compaction threshold, level-7 composition, long documents + the fix in training).
6 Get started: three commands, six lines, the limits sentence, "Full limitations →".
7 Footer band (yellow): tape edge, wordmark, links, the ledger line "every number traces to /site/data".
Research page: the eight findings, full limitations, licences, plus the five explorables (P1–P5).

## 5. Interactive research (explainer)
P1 **Latency explorer**: steppers for state length × K × M over `latency.json.native_sweep`; native vs
   prompting; "ms per extra question" tile. P2 **Calibration explorer**: model × tier, hover a bin to see
   `n/N × |acc − conf|` sum to the ECE. P3 **Architecture trace** (also compact on index): scrub one real
   decision through the forward pass, token counts from the real tokenizer, head formulas cited from
   `native.py`. P4 **Run explorer**: 134 runs × 75 eval sets, pivot by set or by run, floors where
   sourced, released runs marked, the bug/verdict timeline as the date axis. P5 **Training curves**:
   `train/loss` and `val/*` for the 66 W&B runs, aligned by step, pinned comparisons (ts1 vs ts1b, bs16
   vs cand, the 4B/8B/14B ladder). Data: `scripts/export_wandb.py` → `site/data/curves/*.json`;
   `precompute_research.py` → `runs-index.json`, `trace.json`. Hand-rolled SVG; every chart prints its
   source line; unreleased runs hollow.

## 6. Motion (GSAP 3 + ScrollTrigger, CDN)
Hero once per session, 880 ms total: nav fade → display lines rise (stagger 70 ms) → the headline box
fills to the argmax while its value counts → sub + pills → the tape wipes in and starts running (one
column per 2 s). On scroll, once: highlight boxes fill, stat numbers count up while their box fills,
decision-card bars fill with a 40 ms stagger (∅ last), register rows fade. DOOM/Snake card pins for two
viewports with a scrubbed readout (desktop only). Hover: pills and links only. Reduced motion: final
states, static tape, no pin. Never animate layout properties or the yellow itself.

## 7. Tech
- Tailwind v4 `@theme` tokens in `site/src.css` → `site/style.css` via `@tailwindcss/cli` (devDependency,
  `npm run build:css`); utilities for layout, components in `@layer components` so the HTML stays
  readable. No framework; ES modules; GSAP from CDN; hand-rolled SVG charts; games/inference unchanged.
- Keep: `server.py`, `api.js` (queued, replay fallback), `bars.js` (becomes the fill-to-p renderer),
  engines and renderers, `record_*` scripts, data JSON with sources. Drop: dots.js, ink.js, charts.js'
  old palettes (recolour), research.css.
- Client rules carried over: results before demos; no scroll containers anywhere; every number with its
  control; prose through stop-slop + humanizer; sync from the research branch before any numbers change.

## 8. Build order and checkpoints
1. Tokens + type + `.p-box`/`.nul` + pill + decision card + tape → **hero mock, screenshot, sign-off**.
2. Index: hero, results, how (with the compact trace), demos (DOOM, Snake, rules, Try it, rows), fails,
   start, footer. 3. Motion pass. 4. Research page + P1–P5 (P5 after the W&B export). 5. Fresh-eyes
   review (10-second test, brief compliance, overflow audit, mobile). 6. Copy pass.
