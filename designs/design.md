# Typical — design direction (v10)

Lane: **Browserbase's "broadsheet meets data terminal"** (light canvas, oversized geometric display, one signal colour as typographic punctuation, pastel grouping, mono metadata, black pills, no shadows). We keep the grammar and change every noun: yellow instead of orange, a squared instrument face instead of GT Planar, a data field instead of a pixel mountain, and one device Browserbase does not have: **the box is the number**.

Note: this supersedes the "Atoms" obsidian reference in PRODUCT.md §Style; the client rejected v8 (dark) and v9 (light, cards) as undesigned. Everything below is derived from one idea, not from a theme.

## 1. Concept

**Ink fills to p.** Typical's whole personality is a probability with an explicit abstain, so every decision on the site is drawn the same way: a rectangle whose fill, from the left, covers exactly p. The winning candidate is the fullest box; ∅ is the dashed empty box. In a headline the signature highlight box is this box at p = 1 (a full yellow mark); in the hero it *fills on load to the live winning probability*; in the results, `.804` sits inside a box filled to 80.4%; in a decision card the candidate labels are the bars. One device, every screen, always backed by a real number. The generated imagery is the same device at scale: **the tape**, a field of 1,112 recorded decisions from `site/data/replays.json`, one column per decision, one cell per candidate, cell fill = p, argmax cell in yellow, ∅ dashed. It reads as a departures board that is currently running (the brand's voice words: machined gauge, dot-matrix board, lab notebook). Not Browserbase because: hue 92° not 35°, data columns not pixel mountains, Technor not GT Planar, and our highlight box carries a value instead of decorating a phrase.

## 2. Palette (Restrained: tinted neutrals + one signal)

| Token | Hex | OKLCH | Role |
|---|---|---|---|
| `--color-canvas` | `#FAF8F1` | `oklch(0.979 0.009 94)` | page, cards on pastel |
| `--color-ink` | `#17150F` | `oklch(0.196 0.012 92)` | text, pill fill, hairlines on pastel, tape cells |
| `--color-grey-1` | `#5E5A50` | `oklch(0.468 0.017 89)` | muted body, captions (6.5:1 on canvas) |
| `--color-grey-2` | `#D8D3C6` | `oklch(0.867 0.018 89)` | hairlines, empty bar track, tape empty cell |
| `--color-straw` | `#F4E8BF` | `oklch(0.930 0.055 94)` | decision-card surface (signal's tint; "this is a decision") |
| `--color-sage` | `#DDE7D3` | `oklch(0.915 0.029 129)` | code / model-window surfaces |
| `--color-fog` | `#E5E6E0` | `oklch(0.922 0.008 114)` | register table stripes, quiet panels |
| `--color-clay` | `#EFDCCB` | `oklch(0.905 0.031 64)` | limits / "what does not work" surfaces |
| `--color-signal` | `#F2C400` | `oklch(0.837 0.171 92)` | **Board Yellow**, the only chroma |

Signal rule: yellow appears only (a) inside a fill-to-p box behind ink text, (b) as the argmax cell of the tape, (c) as the footer band. Never text, never button, border, icon, hover, or gradient. Ink on signal is 11:1.

Why yellow: Browserbase owns hot orange (h 35), typesafe.ai is `#1E1E1E` on `#FEFEFE` with salmon `#f386a1` / magenta `#d45bb6` and a pixel terminal face (LisaTerminal) plus JetBrains Mono; Hume violet, Together periwinkle (h 285), Runway cobalt, OpenWeb electric blue. Yellow at h 92 is 57° from orange and on nobody's board, it is literally a highlighter on a lab notebook and the LED of a split-flap departures board, and it is the one hue where black text stays legible at full chroma, which the fill-to-p device needs (text straddles filled and unfilled halves).

## 3. Typography

- **Display: Technor** (Fontshare, free for commercial use), weights 400/500. Squared geometric with even widths, reads like engraved instrument lettering; lowercase stays calm at 400. Has `salt` alternates (evaluate for a/g during build, default off). Load: `<link rel="stylesheet" href="https://api.fontshare.com/v2/css?f[]=technor@400,500&display=swap">`. Display only at ≥ 34px.
- **Body: Host Grotesk** (Google, OFL), 400/500. Neutral, tall x-height, disappears next to Technor. `family=Host+Grotesk:wght@400;500`.
- **Mono: Martian Mono** (Google, OFL), `wdth 87.5`, 400. Wide-cell, machined; the readout voice for every number, eyebrow and label. `family=Martian+Mono:wdth,wght@87.5,400`. Set `font-variant-numeric: tabular-nums` everywhere numbers move.
- **∅ is never a font glyph** (none of the three carry U+2205). It is `.nul`: a 1em circle, 1.5px dashed ink border, one rotated 1.5px diagonal. It is also the favicon and the mark beside the wordmark.

One combined Google link: `https://fonts.googleapis.com/css2?family=Host+Grotesk:wght@400;500&family=Martian+Mono:wdth,wght@87.5,400&display=swap`.

| Role | Face | Size | lh | tracking |
|---|---|---|---|---|
| caption | mono | 12px | 1.3 | +0.06em, uppercase |
| body | Host | 16px | 1.55 | 0 |
| subheading | Host 500 | 22px | 1.3 | −0.01em |
| heading | Technor 500 | 34px | 1.1 | −0.02em |
| heading-lg | Technor 500 | 56px | 1.02 | −0.03em |
| display | Technor 400 | `clamp(64px, 9vw, 136px)` | 0.95 | −0.04em |

Ratio ≥ 1.27 between steps. Body measure 64ch.

**Highlight box (`.p-box`)**: `display:inline-block; padding:0 .1em; border-radius:2px; position:relative; isolation:isolate`. `::before` = signal fill, `inset:0; width:100%; transform:scaleX(var(--p,1)); transform-origin:left; z-index:-1`. For p < 1 add `box-shadow: inset 0 0 0 1.5px var(--color-ink)` so the unfilled remainder reads as "the rest of the mass". Same font, size and baseline as the surrounding text; one box per headline; `--p` is always set from data (hero: live decision's argmax, stats: the number itself).

## 4. Tokens: Tailwind v4 `@theme`

Pick **Tailwind v4 `@theme` in `site/src.css`, standalone CLI, no config file, no PostCSS, no Node runtime** (`tailwindcss -i site/src.css -o site/style.css --minify`). Reasons: `@theme` variables *are* CSS custom properties, so ES modules and GSAP read them with `getComputedStyle` and animate `--p`; SCSS maps would duplicate every value into a compile-only namespace and need a second toolchain. Utilities only for layout; every component is a class in `@layer components`.

```css
@import "tailwindcss";
@theme {
  --color-canvas:#FAF8F1; --color-ink:#17150F; --color-grey-1:#5E5A50; --color-grey-2:#D8D3C6;
  --color-straw:#F4E8BF; --color-sage:#DDE7D3; --color-fog:#E5E6E0; --color-clay:#EFDCCB; --color-signal:#F2C400;
  --font-display:"Technor",ui-sans-serif; --font-body:"Host Grotesk",ui-sans-serif; --font-mono:"Martian Mono",ui-monospace;
  --text-caption:12px; --text-body:16px; --text-subheading:22px; --text-heading:34px; --text-heading-lg:56px;
  --text-display:clamp(64px,9vw,136px);
  --spacing:4px;  /* 4px base: use spacing-1…-56 = 4…224 */
  --radius-card:4px; --radius-tag:999px; --radius-button:50px;
  --width-page:1200px; --width-prose:64ch; --width-band:1440px;
  --z-field:0; --z-content:1; --z-nav:50; --z-overlay:100;
  --duration-tap:120ms; --duration-state:240ms; --duration-reveal:480ms; --duration-enter:800ms;
  --ease-out:cubic-bezier(.25,1,.5,1); --ease-out-expo:cubic-bezier(.16,1,.3,1); --ease-in-out:cubic-bezier(.65,0,.35,1);
  --breakpoint-sm:40rem; --breakpoint-md:48rem; --breakpoint-lg:64rem; --breakpoint-xl:90rem;
}
```

Spacing scale in use: 4 8 12 16 24 32 48 64 96 160 224. Section gap 96 (160 before the tape sections). Card padding 24. Hairline = 1px `grey-2` on canvas, 1px `ink` on pastel. No shadows anywhere. Tape cell pitch 8px (6px on mobile).

## 5. Components

- **Nav**: 56px, canvas, 1px grey-2 bottom, sticky. Left: `.nul` mark + "Typical" in Technor 500 20px. Centre: Results / Demos / Research / Code, body 16px, 24px gap. Right: "Hugging Face →" ghost pill, "Try it live" primary pill. Mobile: mark + primary pill + a mono "MENU" that opens a full-canvas list.
- **Pill button**: primary = ink fill, canvas text, Host 500 16px, `padding 8px 20px`, height 36, radius 50. Ghost = 1px ink border, transparent. Icon (→) to the right, 6px gap. Hover: primary fill → `oklch(0.30 0.012 92)`, ghost bg → straw, 120ms. Focus: 2px ink outline, 2px offset.
- **Highlight box**: §3.
- **Section header**: left-aligned, not centred. Mono eyebrow (caption, uppercase, grey-1) → headline (heading-lg, one `.p-box`) → sub (subheading 400, grey-1, ≤ 64ch). 16px / 24px gaps.
- **Stat tile**: canvas, 1px grey-2, radius 4, padding 24. Row 1 mono caption ("JEVBENCH STANDARD · SMALL / MEDIUM"). Row 2 the number in Technor 56px inside a `.p-box` with `--p` = the value (`.694` box filled to 69.4%). Row 3 body 14px grey-1 control/floor line ("majority .311, n = 72, SE .058") and a mono source path. The floor is mandatory (PRODUCT §2).
- **Decision card**: straw surface, radius 4, padding 24, 1px ink border. Mono header: `CHOICE · RECORDED` left, `212 MS · MPS` right. State text (body). Question (mono 14). Candidate rows: label inside a `.p-box` whose `--p` = prob, value in mono to the right; argmax row Host 500. Last row `.nul` + dashed track, `--p` = p_null. Footer mono: model · pass count · source.
- **Register table**: full-width, no vertical rules, 1px grey-2 row rules, fog stripe on alternate rows. First col Host 500, numbers Martian tabular; every metric cell has a floor cell beside it in grey-1. Head row mono caption uppercase.
- **Feature card + model window**: sage surface, radius 4, padding 24, 4-col grid (2 on md, 1 on sm). Inside, the **model window** replaces Browserbase's browser mockup: canvas rectangle, 1px ink, 3px radius, top bar is a mono strip `state · question · K=4 · ∅` instead of traffic lights, body is a miniature decision card (3 rows + ∅). Headline Technor 22px below, body 16 grey-1, ghost pill.
- **Code card**: sage, radius 4, mono 14px/1.5, one white input-style block per snippet, 1px grey-2, copy ghost pill top-right. Output lines are decision rows using `.p-box`.
- **Demo card**: ink surface (the only dark surface; the game canvas needs it), radius 4, 1px ink, canvas 16:10 top, readout panel below on straw: mono rows of the live distribution for the current frame (label `.p-box` bars), measured ms and device printed exactly as measured (content-spec §1 rule 14).
- **Footer band**: signal fill, ink text, full-bleed. Top edge is one row of the tape in ink cells. Columns: wordmark 56px, links in body, mono claim-ledger line ("every number on this page traces to a file: /site/data").

## 6. Motion (GSAP 3 + ScrollTrigger via CDN; `gsap.matchMedia` for reduced motion)

Hero load, once per session (`sessionStorage.typical_hero`), total 880ms:
- 0ms nav opacity 0→1, 120ms.
- 80ms display lines: `y:16→0, opacity:0→1`, stagger 70ms, 600ms, ease-out-expo.
- 420ms hero `.p-box` `--p` 0→argmax (scaleX), 480ms, ease-out-quart; mono value beside it counts up in step.
- 520ms sub + pills fade, 240ms.
- 600ms the tape reveals left→right with `clip-path: inset(0 100% 0 0 → 0 0 0 0)`, 280ms; then it runs: one new column every 2s from the live queue, `x` shift by one pitch via transform.

Scroll (all `once:true`, `start:"top 80%"`): highlight boxes fill 480ms ease-out-quart; stat numbers count up 700ms with tabular nums (snap to 3 decimals) while their box fills; decision-card bars fill with 40ms stagger, ∅ last; register table rows fade 240ms, stagger 30ms, cap 10 rows. Pinned demo scene: the Snake/DOOM demo card pins for 2 viewports, `scrub:0.6` steps the readout through 24 recorded decisions, canvas plays the matching replay frames; on md and below no pin, a play button instead. Hover: pills 120ms colour; links underline 1→2px thickness; stat tiles nothing; tape cells nothing (it is data, not a toy).

Easings: enter ease-out-expo, state ease-out-quart, toggles ease-in-out. No bounce, no elastic, no blur reveals. Exits 75% of enters.

Reduced motion: `gsap.matchMedia("(prefers-reduced-motion: reduce)")` → everything renders in its final state, count-ups print the final number, tape static, no pin, opacity-only 150ms fades allowed.

Never animate: width, height, top/left, margin/padding, font-size, letter-spacing, `border-width`, grid/flex tracks, `background-position`, the signal colour itself (no colour tweens through yellow), and no parallax on the tape.

## 7. Imagery language

All imagery is generated in the browser from `site/data/replays.json` (810 replays, 1,112 decisions, K = 2…150, p_null recorded). **The tape**: a `<canvas>` band, 8px cells, one column per decision (sorted by recording order), rows = candidates by descending p, cell = square filled with ink at alpha p (quantised to 5 levels so it stays crisp), argmax cell solid signal, ∅ cell dashed grey-2 outline whose fill = p_null. Columns with K > 8 show top 7 + "…" row. A mono caption under it always states the count and the source path. Where it appears: hero (full-bleed band under the headline, 120px tall, running), section dividers (one 24px row showing only argmax p as fill height, static), demo readouts (one column per frame), footer top edge (ink cells on signal). Never: stock photos, 3D renders, gradients on chrome, dither/ASCII costume, random noise cells, decorative dots without a value behind them, more than one tape per viewport.

## 8. Wireframes

Hero 1440:
```
┌────────────────────────────────────────────────────────────────────────────┐
│ (∅) Typical      Results  Demos  Research  Code        [Hugging Face →] [Try it live] │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  SMALL OPEN DECISION MODELS · 1.7B / 4B · OPEN WEIGHTS         (mono caption)│
│                                                                            │
│  One pass, a ▓▓▓▓▓▓▓▓▓░░░░ 0.58                                            │
│  ▲ .p-box fills to live argmax   probability                               │
│  for every option, and an explicit (∅).          Technor display, 3 lines  │
│                                                                            │
│  Typical reads text once and answers typed questions      [See results]     │
│  as calibrated probabilities. 45 ms per decision.         [Read research]   │
│                                                                            │
├─ THE TAPE ─────────────────────────────────────────────────────────────────┤
│ ▓░▓▓░▓░░▓▓▓░▓░▓▓░░▓▓▓░▓░░▓▓░▓▓░░▓▓▓░▓░▓▓░░▓▓░▓░░▓▓▓░▓░▓▓░░▓▓▓░▓░░▓▓░▓▓░░▓▓ │  120px, running
│ ░░░▓░░▓░░░▓░░▓░░▓░░░▓░░▓▓░░░▓░░▓▓░░░▓░░▓░░▓▓░░░▓░░▓▓░░░▓░░▓░░▓░░░▓░░▓▓░░░ │
│ ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ │  ∅ row
│ 1,112 RECORDED DECISIONS · SITE/DATA/REPLAYS.JSON · NEWEST RIGHT           │
└────────────────────────────────────────────────────────────────────────────┘
```

Hero 390:
```
┌──────────────────────────┐
│ (∅) Typical   [Try it]  MENU │
├──────────────────────────┤
│ SMALL OPEN DECISION      │
│ MODELS · 1.7B / 4B       │
│                          │
│ One pass,                │
│ a ▓▓▓▓▓░░░ 0.58          │  display 64px, 5 lines
│ probability              │
│ for every option,        │
│ and an explicit (∅).     │
│                          │
│ Reads text once, answers │
│ typed questions as       │
│ calibrated probabilities.│
│ [See results]            │
│ [Read research]          │
├──────────────────────────┤
│ ▓░▓▓░▓░░▓▓▓░▓░▓▓░░▓▓▓░▓░ │  tape 72px, 6px cells
│ ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌ │
│ 1,112 DECISIONS · SOURCE │
└──────────────────────────┘
```

Feature row (two-column API block, 55 / 45):
```
┌──────────────────────────────────────┐  ┌───────────────────────────────┐
│ sage card                            │  │ READ ONCE, ASK MANY  (caption) │
│ ┌ state · question · K=4 · ∅ ──────┐ │  │                                │
│ │ "I was charged twice…"           │ │  │ One KV pass,                   │
│ │ Which team owns this ticket?     │ │  │ ▓▓▓▓▓▓▓▓ 2.7 ms per            │
│ │ billing    ▓░░░░░░░░░ .05        │ │  │ extra question.   Technor 34   │
│ │ shipping   ▓▓▓▓░░░░░░ .35        │ │  │                                │
│ │ support    ▓▓▓▓▓▓░░░░ .58  ←500  │ │  │ Body 16 grey-1, ≤ 64ch, with   │
│ │ (∅)        ╌╌╌╌╌╌╌╌╌╌ .08        │ │  │ the H2 numbers and their       │
│ └──────────────────────────────────┘ │  │ K=2/32/256 caveat.             │
│ 212 MS · MPS · TYPICAL-SMALL         │  │ Read the latency table __      │
└──────────────────────────────────────┘  └───────────────────────────────┘
```
