# DESIGN.md — Typical, "research console on paper"

Sources: designs/TOGETHER_AI_DESIGN.md (skeleton), designs/HUME_DESIGN.md (data bars, pastel tiles),
designs/RUNWAY_DESIGN.md (image-card grid, restraint). Approved mock: site/mock-paper.html.

## Color
paper #ffffff canvas · bone #fff9f3 section wash and card surface on white · ink #0a0a0a text ·
slate #4d4d4d secondary · smoke #7a7876 muted · hairline #d6d6d6 · midnight #010120 (ONE dark band per
page; white text on it, never pure black) · periwinkle #bdbbff small punctuation only (dashes, active
underline) · pastel tiles: sky #c1dff9, blush #fde3f6, peach #ffdccd, mint #c8f6f9 (category-coded, never
decoration) · violet #c094e4 = every probability bar. No gradients on chrome, no shadows, 4px radius on
cards/buttons/inputs, 12px on the hero decision card only.

## Type
Inter Tight (stand-in for The Future): 400/500. Display 56px −1.7px lh 1.1 (bold line + light slate
continuation line); h2 40px −0.8px; body 16px −0.16px lh 1.4; 18px ledes; 14px captions.
JetBrains Mono (stand-in for PP Neue Montreal Mono): 11px 500 uppercase labels/eyebrows/badges, 13px
buttons, 12px question text inside decision cards. Numbers tabular.

## Components
- Buttons: primary = ink fill, white mono 13px, 4px, 8px 16px; ghost = hairline border. One primary per view.
- Stat tile: pastel fill, mono label with ↑/↓, 64px number, 14px slate caption. No border.
- Decision card: bone surface, 12px radius, 28px padding: eyebrow mono (type · source · ms · device),
  state text 16px, question mono 12px, rows label + violet bar with value (value inside when p ≥ .15,
  outside in slate otherwise), ∅ row dashed outline. This IS the readout everywhere (demos, Try-it, games).
- Register: hairline table, mono 11px headers, ink rule under the header row, tabular numerals; stacks to
  label/value rows ≤ 600px.
- Demo card (Runway): white card, hairline border, 4px; canvas/iframe on top (16:10 or 480px), mono label
  row (title · tag · measured ms), decision rows below. Two-up grid for games; exhibits open inline as a
  full-width bone panel under an index row.
- Research band: midnight, cards #0b0b33 with a 3px periwinkle dash, mono category, 22px title, mono meta.
- Nav: white, brand (violet dot + wordmark 18px 600), links 16px, ghost + primary CTA. Sticky, hairline
  bottom only.

## Layout
1200px max, 32px gutters, sections 72–96px apart, tiles/cards 16px gaps, 4px base grid. Hero two-column
(copy 52% / decision card 48%). Results: three tiles → register → charts (violet marks, hairline grid,
mono axis labels) → disclosure in a bone box. Demos: two-up game cards, then an exhibit index. Research
band → Get started (bone wash) → footer. Nothing scrolls inside the page except game canvases.

## Motion
None on load. Live data changes ease (bars 400 ms ease-out-quint). Hover: ghost → ink text.
