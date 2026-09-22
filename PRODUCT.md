# PRODUCT.md — Typical

register: brand

## What it is
Typical is an open family of small decision models (1.7B and 4B, Qwen3 backbones). Give it text and a typed
question, it returns one probability per candidate plus an explicit abstain (∅), in one forward pass. No
generation, no parsing. 45 ms per decision on an H100; a few ms per extra question on a cached state.
Public weights and an inference package on Hugging Face (OzLabs/typical-small, typical-medium).

## Users
ML engineers and researchers arriving from Hacker News, a paper, or a colleague's link. They have 60 seconds,
they distrust marketing, and they respect a page that shows its controls and its failures. Secondary: a
technical founder deciding whether "System One" models are worth a prototype.

## Brand
The product's whole personality is the probability. It does one thing: it decides, and it says how sure it
is. Voice words (physical objects): a machined gauge, a dot-matrix departures board, a lab notebook.
Calm, exact, a little dry. The page should feel like an instrument that is currently running, not a brochure.

## Anti-references
- Generic dark dev pages: boxes, tables, small grey type, a purple gradient.
- Jev / typesafe-style claims ("193× faster", "zero hallucinations"). We publish controls, floors, and
  disclosures next to every number.
- Editorial-magazine aesthetics (display serif, italics, rules). Rejected by the client.
- Thermal/ASCII/dither costume. Rejected by the client.
- Renaissance gallery on beige. Rejected by the client.

## Strategic principles
1. Results before demos; a visitor knows what it is and how good it is in ten seconds.
2. Every number on the page traces to a file; every claim has its control beside it.
3. One dominant idea per screen. Black space is a material, not emptiness.
4. Demos are real (real DOOM in WASM, Three.js road, canvas Snake) and playable; the model's decisions are
   visible as they happen.
5. Honesty is the brand: what fails is on the page, tagged.

## Style reference (client-supplied, binding)
`designs/BROWSERBASE_DESIGN.md` is the anchor lane (editorial broadsheet meets data terminal: light
canvas, oversized geometric display type, one signal colour used as a typographic highlight and a
footer band, pastel grouping surfaces, mono metadata, black pill CTAs, no shadows, digital-native
imagery). Not a copy: our own signal colour, faces, and imagery. The other files in `designs/`
(Hume, Together, Runway, OpenWeb) are secondary references for restraint and data presentation.
The working brief is `PLAN5.md`; the token system lives in `site/src.css`.
