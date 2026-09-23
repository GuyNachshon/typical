# Typical: marketing strategy (CMO, 2026-09-22)

Sources: PRODUCT.md, content-spec.md §1 (ledger), site/data/models.json, frozen.json, demos/spec-v4.md, designs/BROWSERBASE_DESIGN.md. Market: typesafe.ai ("System One models", "193.6x faster, 244.6x cheaper", "zero hallucinations"), benchmarkheaven.com/jev-models ("decision models: state and a bounded rubric in, a typed answer out"), jevai.org/apps ("very very fast, and very very cheap"). Reference sites: Resend ("the email API for developers"), Linear ("the product development system for teams and agents"), Raycast ("your shortcut to everything"), Vercel ("Agentic Infrastructure"). All four state the category noun in the first line and put the product on screen before any pitch.

## 1. Positioning and category

**Statement:** Typical is the open decision model: text state and a typed question in, a probability for every candidate and an explicit ∅ out, in one forward pass, with the control printed beside every number.

**Category noun: "decision model".** Keep it. The market (benchmarkheaven, Jev) already uses it; fighting a noun costs a paragraph on every page and we have 60 seconds. Rejected: "System One model" (typesafe's phrase, we would be quoting the incumbent), "decision head" (accurate but reads as a component, not a product), "typed classifier" (undersells runtime candidates and ∅), "probability model" (means something else in statistics). We own the modifier, not the noun: **open** decision model, and the posture "shows its floors".

**Standing next to Jev without numbers.** Never name Jev on index.html; the research page may show leaderboard neighbours as context only. Differentiate on axes a hosted API cannot occupy: weights on Hugging Face, runs on your GPU or laptop, candidates defined at call time, ∅ as a first-class output, every number with its floor and disclosure. Their headline is a multiplier; ours is a measured millisecond with its conditions. That contrast is the positioning; we never have to say their name.

## 2. Audience jobs-to-be-done

| | 10 s | 60 s | 5 min | What earns trust |
|---|---|---|---|---|
| **ML engineer evaluating** (HN, paper link) | Size (1.7B/4B), latency with conditions (45 ms, H100, K=2), one accuracy with its floor | Results table with floors; JevBench with D1 in the same viewport; reliability diagram | Research page, `inference/` code, six lines, run it on their machine | The hard tier printed at chance (.432/.423, chance .336, SE .058) on the front page |
| **Founder deciding on a prototype** | "You define the candidates; it returns a probability per candidate plus ∅; one pass" | Playground (one ticket, four questions, total ms vs M× single) and rules-flip demo | Get-started, latency-vs-K curve, Limitations list | Demo captions that state what failed and the measured local ms with device |
| **Researcher checking rigor** | Backbone, tap depth (20/28, 26/36), typed heads, factored ∅ | Eight findings each with a matched control; frozen-backbone control table (1.7B .528 → .694 std; 4B .778 → .806) | REPORT sections, 73 runs in 6 days timeline, licence section on the card | Negative results on the page: candidate-blind states, preview .750 → .694 for calibration and typed heads |

## 3. Message hierarchy

**Hero headline options** (highlight box = [bracketed] phrase, one per line):
1. Small open models that [decide]. (current h1; recommended)
2. A probability for every option, and [∅ when it should not answer].
3. Read the state once. Answer [typed questions] with probabilities.
4. Not a chatbot. [A decision], in one forward pass.
5. Your candidates, its probabilities, [45 ms].

**Sub-headline:** Text state and a typed question in. One probability per candidate plus an explicit ∅ out, in one forward pass: 45 ms on one H100, a few milliseconds per extra question on the same state. 1.7B and 4B weights on Hugging Face.

**Three proof points (hero tiles, ledger H1/H2/H3/H5):**
1. 45 ms per decision (1.7B, H100, K=2); 2.7 ms per extra question on a cached state.
2. 150 runtime-defined intents: .804 / .847 (small / medium), out-of-scope abstain recall .827.
3. Same facts, different rule, different answer: held-out rule families .836 / .874; rubric flip both-correct .718 / .743.

JevBench (.694 / .806 standard) goes in Results, not the hero: it must sit within one viewport of D1, and the hero cannot carry D1.

**Objections to pre-empt, in our words first:**
- *Hard tier at chance.* Say it in the Results intro, with the SE, before the reader finds it. Over-confident there (ECE .26 / .28).
- *Text-only.* Doom and Snake read pre-computed sentences; the engine renders and presses keys. Caption says so on the plate.
- *Not an LLM.* No generation, no arithmetic, no timezones, no trade-offs, states ≤ 1,024 tokens. Level-7 composition at chance.
- *Order.* Noul is exactly order-invariant; Choice is order-robust (reorder Δp .01–.08), never "invariant".
- *Two sizes, three checkpoints.* Never "three sizes". Training code not yet public; say "to follow".
- *Local ms is not H100 ms.* Every live result shows its own ms and device.

## 4. Narrative arc (8 beats)

1. **Hero.** Headline, sub, three tiles. Purpose: what it is and how fast, in ten seconds.
2. **Decision card.** One real decision rendered as dots: state, question, candidates, ∅, probabilities. Purpose: show the output before explaining it.
3. **Results.** Table with floors, JevBench with D1 inline, reliability diagram, frozen-backbone control. Purpose: how good, with the failures tagged. Results before any demo (client rule).
4. **Three primitives.** Choice, Noul, Score; one head, one cached state. Purpose: the API shape.
5. **Demos, practical first.** Playground, rules flip, inbox K=150, abstain, then Snake and Doom last with their honest captions (61/62 boards, chance .48; 200/200 E1M1 decisions, 4 kills). Purpose: it is running, and you can see it decide.
6. **Get started.** pip line, six lines of Python, HF links. Purpose: remove the gap between reading and running.
7. **Limitations.** The MUST-NOT list in first person, tagged. Purpose: trust by disclosure.
8. **Research teaser and footer.** Eight findings with controls, 73 runs in 6 days, licence link to the card. Purpose: the rigor persona's door.

## 5. Voice: five rules

1. **A number never travels alone.** Good: "PagerDuty .817, constant-prediction floor .792." Bad: "82% accurate on real incident triage."
2. **What it does not do sits in the same sentence.** Good: "It trends toward the food; it does not plan." Bad: "Watch Typical master Snake."
3. **Only verbs the data backs.** Allowed: reads, returns, abstains, measured, held-out, at chance. Banned: unlock, supercharge, blazing, reasoning, understands, beats, state-of-the-art, human-level.
4. **No triads, no em dashes, no rhetorical questions.** Good: "45 ms on an H100. Open weights." Bad: "Fast, calibrated, and open."
5. **Conditions on every latency.** Good: "62 ms, M4 Pro, live." Bad: "real-time".

Also: three decimals as in source, sentence case, no "simply", "seamlessly", "we're excited".

## 6. CTAs

| Persona | Primary | Secondary |
|---|---|---|
| ML engineer | "Weights on Hugging Face" (OzLabs/typical-small, typical-medium) | "Results and controls" (jump to beat 3) |
| Founder | "Run it locally" (copies the pip line) | "Open the playground" |
| Researcher | "Research page" | "Findings with controls" |

**What "Try it" means.** The site never says "Try it". Live mode needs the local server (`localhost:8787`, `OzLabs/typical-small` on Apple silicon); when the site detects it, buttons read "Run" and results show measured ms plus device. Otherwise buttons read "Replay" and play the precomputed outputs, labelled as such with their recorded ms. The primary founder CTA is therefore the pip line, not a hosted sandbox we do not have. If a hosted playground ships later, it must show its own device and ms, never the H100 tile.
