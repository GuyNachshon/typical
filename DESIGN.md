# DESIGN.md — Typical, "the dot is the probability"

## Concept
One visual system carries the whole site: a dot matrix in which every dot's brightness is a real probability
the model produced. The wordmark, the headline numbers, the charts, the decision readouts and the game HUDs
are all the same dots. It is Nothing's dot-matrix brand system pointed at our data, on the Atoms canvas.
Nothing on the page is decorative; if a dot is lit, a number made it so.

## Color (Restrained, Vercel-black reference, warm-tinted)
- canvas `oklch(0.10 0.006 80)` (≈ #0c0b09; not pure black)
- cream `#fff7dd` text and hairlines (hairlines at 22% alpha)
- champagne `#c8ad86` the only accent: lit dots, tags, the winning candidate, hover
- ash `#66635f` muted text and unlit dots (at 35% alpha)
- DOOM's own palette is allowed inside its canvas; everything else is the four above.

## Type
Switzer (Fontshare, free) 400/500. Scale: 10 caps (+0.18px), 12, 14 body (line-height 1.5 on dark),
16 (−0.13px), 44 headline (−1.85px, 1.13). No other sizes. Numbers use `tabular-nums`. Dot-matrix numerals
(5×7 dots) are the display size: 8–14px dots, so a "45" can be 120–200px tall without a new type size.

## Dot primitives (`js/dots.js`)
- `dotText(el, text, {size, values})`: 5×7 dot font; each dot's alpha from `values[i]` (a probability) or,
  when no data applies, a fixed 0.9. Champagne when lit ≥ .5, cream below, ash off.
- `dotBars(el, rows)`: one row per candidate: a 24-dot strip filled to p, label 12px, numeral tabular 14px.
  Winner row's dots champagne; ∅ row outlined dots.
- `dotChart(el, series)`: bars/points drawn as dots on a dot grid; axes as ash dots; labels 10px caps.
- `dotField(el, values)`: the hero wall, N×M dots.
Static on load. Live updates (a decision arrives) change dot alpha with a 120 ms ease-out; nothing moves.

## Layout
1200px column, but the hero wall and the game plates bleed to the viewport. One dominant element per
screen; sections separated by a single hairline and 80–120px. Left-aligned text, asymmetric plates
(a game 60% / its readout 40%), no card grids. Tables are hairline registers with tabular numerals.
Tags are 100px pills. No scroll containers anywhere; content folds inline.

## Motion
None on load. Live-data alpha changes only. Hover: cream → champagne, 120 ms.

## Screens
0 nav · 1 hero wall (dot-matrix TYPICAL, live) + 44px headline + Results → · 2 results: three dot-matrix
numerals, register, dot charts, disclosure · 3 how it works: Choice/Noul/Score as dot glyphs + code ·
4 demos as plates: DOOM, Drive, Snake, then the exhibits as an index that opens inline · 5 limits (short) ·
6 install · footer.
