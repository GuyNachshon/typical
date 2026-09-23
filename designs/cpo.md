# Typical site — CPO strategy (2026-09-22)

Inputs: `PRODUCT.md`, `.context/content-spec.md` §1, `site/data/demos/spec-v3.md`/`spec-v4.md`, `site/index.html`, `designs/BROWSERBASE_DESIGN.md`, shots `v8-full.png`/`v9-full.png`; browserbase.com, wisprflow.ai (Awwwards SOTD), "AI in Design Report 2026" (SOTD, Developer Award).

## 1. Job and flows

**Job:** In 60 seconds, convince a sceptical ML engineer that Typical is a real, honestly measured decision head (typed question → probabilities + ∅) worth a `pip install`.

**Evaluate** — 1. Hero headline + proof strip (`#top`): what it is, three numbers with floors. 2. Hero decision card: watch it decide (Snake tick → bars). 3. `#results`: table, scaling ladder, latency-vs-K, reliability, D1. 4. `#how`: three primitives + six lines of code + "what it is not". 5. `#fails`: five lines.
**Try** — 1. Hero Snake: press an arrow, take over. 2. `#demos` Doom: play, watch it aim. 3. Rules flip: drag a rule, watch the answer flip. 4. Try it: pick a preset (recorded) or type your own (live).
**Integrate** — 1. `#how` code block. 2. `#start`: three commands + six lines. 3. HF links (nav, footer). 4. `research.html#limitations`.

External patterns adopted: Browserbase — quiet proof strip right after the hero, alternating centred header / two-column product rows, one accent used as a highlight box in the headline, never a button. Wispr Flow — one idea per screen, before/after shown as input-left / output-right, sticky nav with a single CTA. AI in Design Report — charts are editorial objects with one caption, chapter jump-nav, disclosure on demand. What we do not adopt: video heroes, scroll-jacked chapters, logo walls.

## 2. Information architecture (one page + research)

| # | Section | Purpose |
|---|---|---|
| 0 | Nav (sticky) | Results · Demos · Research · Code · **Try it** (pill) · mode pill (`recorded` / `live · MPS`) |
| 1 | Hero | Headline with highlight box on "a probability for every option, and an explicit ∅"; one-line sub; **proof strip** (3 numbers, floor under each); right: Snake board + the decision card for the current tick |
| 2 | Results | Two tables (JevBench / evidence-intent), G1 + G2, G4 standard-tier only, D1 verbatim directly under the JevBench table |
| 3 | How | Three primitive cards, six lines of code, "what it is not" paragraph, ∅ toggle (see §5) |
| 4 | Demos | Doom (wide, primary) → Snake + Drive row → exhibit rows: Rules, Try it, SQL, Doc-20 |
| 5 | What fails | Five one-line items, each with its number |
| 6 | Get started | Three commands, six lines, limitations sentence, link to research |
| 7 | Footer | HF links, "training code to follow" |

**Above the fold, 1440×900:** nav, headline (3 lines), sub, proof strip (45 ms · H100 K=2 / .694 · .806 · chance .311 / .804 · .847 · OOS ∅ recall .827), Snake board + decision card with bars moving. Everything the 10-second rule needs is in this viewport; results tiles are not a second screen.
**Above the fold, 390:** headline (2 lines, 34px), proof strip as one row of three numbers with 10px floors, then the decision card (bars) with a 160px Snake board above it. CTAs below the fold.

**Cut from index → research:** Findings band (8 cards; second results section), G4 hard-tier panels, "In training" status box, Ads exhibit (is_ad .875 vs always-yes .80 — not a marketing demo), Compaction exhibit (fails; keep its one line in §5), chart source captions (become a `title`/hover and one "Sources" line at the end of Results).

## 3. Demo strategy

Ranking (works / instant / fun, each 1–5; public site is recorded-only, so "works" includes working from replay):

| Demo | Works | Instant | Fun | Role |
|---|---|---|---|---|
| Snake | 5 (.984, 216/217) | 5 | 4 | **Hero** |
| Doom (real, WASM) | 4 (200/200, 4 kills; scripted route) | 5 | 5 | **Primary** |
| Rules flip | 5 (.896 → .894; 3/6 flips) | 4 | 3 | **Primary** |
| Try it | 5 live / 3 recorded | 4 | 4 | **Primary** |
| Drive | 5 (70/70) | 3 (7 labels, grey road) | 3 | Secondary (poster + Play) |
| SQL | 4 (3/4 conditions ≥ .96) | 4 | 2 | Secondary (row, opens) |
| Doc-20 | 3 (score acc .33) | 3 | 2 | Secondary (row, opens; show 5 rows then "all 20") |
| ∅ toggle (R6) | 5 (.999 / .000) | 5 | 3 | Comprehension device in §3, not a demo |
| Ads | 2 | 3 | 2 | Cut → research |
| Compaction | 1 | 2 | 1 | Cut → one line in What fails |
| ASCII Doom arena | 4 | 3 | 2 | Fallback only when WASM fails; never listed |

**Hero = Snake**: lightest engine, one-sentence state, 2–3 candidates, fits 390px, replays from `data/replays/snake.json` with no server, and the game is a literal picture of "state → question → probabilities". Doom is the reward at the top of Demos, mounted on scroll, not in the hero (4 MB WASM, keyboard-only, and the first screen must be results, not a game).

**The one pattern every demo shares — the decision card:** `state` (sentence, mono caption "state") → `question` (mono caption "question") → `probabilities` (bars, labels left, value right, argmax filled, ∅ last as a dashed bar) → footer `model · mode · ms · device`. Games render the card beside the canvas; exhibits render one card per row; Try it is an editable card. No demo invents its own readout.

**Live vs recorded:** one word, one place. The eyebrow of every card says `Choice · recorded` or `Choice · live · 71 ms · MPS`; the nav pill mirrors it. Never the word "static" (reads as broken) and never a control that does nothing: in recorded mode Try it shows six preset cards with recorded outputs and one line "Type your own: run the local server (one command)". Live mode enables the textarea and reports its own measured ms and device. Games in recorded mode replay; pressing an arrow switches to "you" for 3 s and the model resumes (already built; keep).

## 4. Interaction and motion

Rule: **only the model's output moves.** Bars tween (150 ms) to the new value — causality: the state changed, the answer changed. The argmax label fills (reward). Game canvases tick (time-driven, 4–6 Hz). The mode pill blinks once on transition to live. Nothing else animates: no scroll reveals, no headline fades, no load animation, no tile counters, charts draw once and stay. Scroll is used only to mount/pause games when in view (one MPS slot). `prefers-reduced-motion`: bars snap, games do not autoplay (poster + Play), everything else unchanged.

Fun is exactly three things: the model visibly deciding (bars every tick), a game you can take over, and a Try it that answers your text.

## 5. Comprehension devices

- The **hero decision card is the diagram**: three mono captions `state / question / probabilities + ∅` appear on the hero card only, once. The visitor learns the pattern from the first example and re-reads it in every demo without labels.
- The **highlight box** in the headline carries the concept sentence: "a probability for every option, **and an explicit ∅**".
- The **∅ toggle** in §How: one premise, a switch "answer in candidates / not in candidates"; recorded p_null .000 ↔ .999. Explains ∅ in one click without a paragraph.
- **Every number wears its floor** inline, same line, ember-ash: `.694 / .806 · chance .311`, `.817 · floor .792`, `45 ms · H100 · K=2`. Never a footnote, never a separate table. D1 stays verbatim under the JevBench table; the hard tier line is the last line of the table, not hidden.

## 6. Mobile (≤800)

Nav links hide; keep brand, mode pill, Try it. Proof strip: one row, three numbers. Results: tables stack; G1/G2 stack; G4 shows the small-standard panel only. Doom and Drive: poster frame + the decision card replaying (no WASM, no Three.js; "play on desktop"). Snake: canvas 160px, swipe to take over. Exhibit rows: accordion, one open at a time, 5 rows visible then "all N". Code blocks scroll horizontally (the only scrolling container allowed). Findings cut, so the page is roughly half the length.

## 7. Risks and the five mistakes to avoid

Mistakes seen in v8/v9:
1. **Nothing measurable in the first screen** (v8: dot mosaic + headline; v9: headline + one card, tiles below the fold). Fix: proof strip in the hero.
2. **Unstyled probability lists under games** (v9: `retreat0.00` text stack, 400px of empty plate). Fix: the shared decision card, always bars.
3. **120-word honest captions on every plate.** Honesty becomes noise. Fix: one stat line (`200/200 rule agreement · 4 kills · chance .20`) + "How it was measured" disclosure.
4. **Exhibits dump every row** (v8: 129 SQL rows, 5,000 px). Fix: show 5–8, then "all N".
5. **Two results sections** (tiles + tables + charts + 8 finding cards) and a `STATIC` pill mid-sentence that reads as an error. Fix: cut Findings, rename modes.

Risks: the Browserbase reference is light/pastel while `PRODUCT.md` binds obsidian black + champagne — the layout patterns transfer, the palette must not; get the client to confirm which before the next build. Doom depends on a 4 MB WASM and keyboard input; if load fails the arena fallback must appear without a broken frame. D1 must stay within one viewport of any JevBench number — verify at 390 after the proof strip moves into the hero.
