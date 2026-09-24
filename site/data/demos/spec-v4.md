# Typical site — demo spec v4: Snake and Doom (2026-09-22)

Supersedes `demo-spec-v3.md` §5 (Doom) and everything earlier about Snake. Same method as v3: `[probe]` against
the live server (`POST http://localhost:8787/api/decide`, `OzLabs/typical-small`, MPS), scripts in
`/tmp/probe4/` (`snake_probe.mjs` builds 62 boards from seeded random walks and probes every variant;
`doom_aim.py` = the 30 v3 Doom situations + 20 new ones with enemies left/right/behind and two 0-ammo crosshair
cases, plus a 24-sentence real-engine robustness set; `snake_sim.mjs` runs the literal rule policy; `route3.mjs`
walks E1M1 waypoints; `calib.mjs`/`ticrate.mjs` measured the wasm build's input timing). The engines
(`site/js/snake.js`, `site/js/games/doom.js`, `site/js/games/realdoom-logic.js`) implement the chosen grammars
verbatim and their self-tests (`node <file>`) pin the exact sentences. Recorders: `scripts/record_games.mjs
--game snake|doom` and `scripts/record_realdoom.mjs` (needs `python3 -m http.server 8788 -d site`).

## What transferred from the Drive fix (v3), and two new rules

- **Pre-compute the comparison, agent-subject, action verb.** Drive's `The car must take the exit on the right`
  pattern is what made Doom aim: `The player must turn left to face the enemy.` fires `turn left` 29/29 where the
  v3 object-location form (`The nearest enemy is 5 cells to the left.`) fired 0/12. Snake: `The snake must move
  right to close the larger gap to the food.` beats the v2 `The food is 3 cells to the right and 2 cells up.`
  (.984 vs .774 on the same 62 boards).
- **New: numbers the model would have to compare itself leak into the decision.** With `The player has 41 health
  and 34 ammo.` in front, an imp in the crosshair reads `retreat` .56 / `shoot` .42 (61 health: .49/.49); drop
  the sentence and `shoot` is .89. 66 of 200 decisions in a recorded run were that failure. The state is now
  *only* the situations that apply; health/ammo live on the HUD.
- **New: the explicit default beats "none of the above".** `explore: applies when the player sees no enemy`
  6/6; `explore: applies when none of the above rules fire` 0/8 (→ shoot/retreat). Any sentence that mentions an
  enemy without being one of the rule conditions (`The player must move forward to reach the enemy.`, `The player
  has an enemy ahead, 4 cells away.`) is read as `shoot` (4/4). So every enemy situation must map to exactly one
  rule: crosshair, turn left, turn right — the arena's 8-heading geometry was changed to make that true (below).
- **Snake: the rule list belongs in the state, not the question.** Four direction rules with no default are off
  the trained 3-rules+default shape: every rules-in-question variant scored .82–.95 with p_null .84–.94, the plain
  question over the same pre-computed state .984 with p_null .04. Kept the plain question; `RULES` is the policy
  the sentences encode (documentation + `greedyPolicy`).
- Confirmed from v3: criteria == full label set and guardrails only for real illegality (Doom: gating the turns on
  an enemy in sight was also 50/50 but cost p_null .13 vs .02); sentence order fixed (Doom is order-robust for the
  hurt sentence, 48/48 either way; Snake is order-robust under the plain question, .984 all three orders, but
  under the rule question unsafe-first drops A-cases 24→16).

---

## A. Snake

**Chosen: S-plain — situations-only pre-computed state, plain question, safe moves as candidates.**

**State template (exact, fixed order; only sentences whose situation holds; A = the move along the axis with
the larger |distance| to the food (tie → horizontal), B = the move along the other axis):**
```
[The snake must move {A} to close the larger gap to the food.]   A safe; "the gap" when B does not exist
[The snake can also move {B} to close the smaller gap.]          B exists and safe
[Moving {d} hits the wall. | Moving {d} hits the snake's body.]  one per unsafe d, canonical order (reversal = body)
```
Examples: `The snake must move right to close the larger gap to the food. The snake can also move up to close the
smaller gap. Moving left hits the snake's body.` / `The snake can also move up to close the smaller gap. Moving
left hits the snake's body. Moving right hits the wall.` / `Moving down hits the snake's body.`

**Question (exact; `QUESTION` in snake.js = `presets.json` `snake.question`, unchanged):**
`Which move brings the snake closer to the food without dying?`

**Candidates:** `candidates()` = `safeMoves()` (no collision, no reversal), canonical order up/down/left/right.
**RULES (the policy the sentences encode; `greedyPolicy` = gold):** close the larger gap (A if safe) → close the
smaller gap (B if safe) → any safe move (first in canonical order). Literal simulation, 300 ticks, 10×10, seeds
7/1/2/3/4/5: 28/27/20/29/24/28 food (dies at 175–300 ticks by trapping itself — greedy has no planning; a
flood-fill guardrail would add 0–6 food and was not worth the explanation).

**Probe, 62 boards (24 A-safe with B, 8 single-axis, 18 A-unsafe/B-safe, 12 neither), gold = literal rules:**

| variant | acc | p_null | per-category (A / A1 / B / DEF) |
|---|---|---|---|
| V0 v2 template (`The head is at column…`, `Safe moves: …`) + plain question | .774 | .00 | 16/24, 5/8, 16/18, 11/12 |
| V1 pre-computed state + 4 direction rules (`right: applies when the snake must move right`) | .887 | .88 | 23/24, 8/8, 17/18, 7/12 |
| V1 rules, `…must move X to close the gap to the food` | .823 | .81 | 19/24, 8/8, 14/18, 10/12 |
| V1 rules, `…moving X closes the larger gap to the food` | .952 | .84 | 23/24, 8/8, 17/18, 11/12 |
| V1 rules, `…must move X or can move X` | .935 | .89 | 23/24, 8/8, 17/18, 10/12 |
| V1 rules, criteria = offered only | .919 | .00 | 24/24, 8/8, 18/18, 7/12 |
| V2 = V1 without the hit sentences | .952 | .89 | 24/24, 8/8, 17/18, 10/12 |
| V3 role labels (`close the larger gap` / `close the smaller gap` / `any safe move`, engine maps) | .871 | .00 | 24/24, 0/8 (wording), 18/18, 12/12 |
| V4 = V1, hit sentences first | .774 | .88 | 16/24, 8/8, 17/18, 7/12 |
| V5 = V1 with `should` for the smaller gap | .790 | .87 | 16/24, 8/8, 18/18, 7/12 |
| **FINAL pre-computed state + plain question** | **.984** | .04 | 24/24, 8/8, 18/18, 11/12 |
| FINAL with `can` instead of `can also` | .952 | .05 | 22/24, 8/8, 18/18, 11/12 |
| FINAL, hit sentences first / must-hit-can order | .984 / .984 | .03 | same |
| FINAL question, all 4 moves offered (no guardrail, diagnostic) | .806 | .00 | 24/24, 8/8, 17/18, 1/12 |
| guardrail-only random (safe moves) | .481 | | |

Chance among offered .48. The one FINAL miss is a DEF board (any safe move is the literal answer; gold = first
canonical).

**Recorded run (`data/replays/snake.json`, seed 7):** 217 decisions, 216 match the rule list (the miss is an exact
.50/.50 tie between the larger- and smaller-gap moves), 24 food, food-ward fraction .926, trapped at tick 217 at
length 27 (recorded as a death frame), mean p_null .06, 62 ms. `record_games.mjs` now records `gold`,
`rule_agreement`, `rule_agreement_literal` (any safe move accepted on the 16 default ticks) and pushes a death
frame when no safe move remains. `render-snake.js`'s scripted policy is `greedyPolicy`.

**Honest caption (on the plate):** "Each tick the board is rendered as the situations that apply, pre-computed as
agent-subject sentences, and the model picks one of the safe moves; the candidate set is the only guardrail.
Following the sentences literally is a greedy policy that eats about 28 food per 300 ticks; the model matched it
on 61/62 scripted boards (chance .48) and on 216/217 ticks of the recorded run, eating 24 food before greedy
trapped it at length 27. It does not plan, and spelling the same rules out in the question made it worse
(.82–.95, p_null .9)."

---

## B. Doom (ASCII arena `doom.js`; real DOOM `realdoom-logic.js`)

**Chosen: A3 — five labels, the model aims, situations-only state with the turn side pre-computed, no numbers.**

**State template (exact, fixed order; only sentences whose situation holds; the lead enemy = crosshair first,
then nearest; real DOOM inserts the monster type):**
```
[The player is badly hurt.]                                                   health < 30
[The player has {n} enemies in sight.]                                        n > 1
{The player has an enemy in the crosshair, {d} cells ahead.                   lead in the crosshair
 | The player has {a/an type} in the crosshair, {d} cells ahead.               (real DOOM)
 | The player must turn {left|right} to face {the enemy|the type|the nearest enemy}.   lead off the crosshair (n>1: "the nearest enemy")
 | The player sees no enemy.}
[A wall is ahead.]
```
"In the crosshair": arena = within ±22.5° (half the 8-heading step, so no turn can do better) and ≤ 8 cells with
a clear line — and `shoot()` now hits exactly that enemy (cone shot, nearest first) instead of the single ray;
real DOOM = `in_crosshair` (|bearing| ≤ 8°). "In sight": arena = ≤ 8 cells, clear line (any bearing; behind →
turn right); real DOOM = `visible` (P_CheckSight), |bearing| ≤ 100°, ≤ 12 cells (768 units — an imp at 18 cells
across the nukage ate 69 pistol shots and all the ammo in a recorded run at 24 cells; 37-cell sightings through the
start room's window cost a turn at 40).

**Question (exact; `QUESTION` in doom.js, shared by realdoom):**
```
Which action applies for the player? Apply the rules in the stated precedence order; the first matching rule wins.
retreat: applies when the player is badly hurt  shoot: applies when the player has an enemy in the crosshair  turn left: applies when the player must turn left to face the enemy  turn right: applies when the player must turn right to face the enemy  explore: applies when the player sees no enemy
```
Criteria dict = `RULES` (order = precedence = label order). **Candidates:** all five, minus `shoot` at 0 ammo
(the only real illegality). Gold = literal first match over the offered labels; with a crosshair enemy and no
ammo nothing fires and the gold is the residual first label (`retreat`); the model answered `turn left` on those
two probe cases, so they are reported separately.

**Engine (`resolve()` / `resolveIntent()` + `keyPress()`):** `shoot`/`turn left`/`turn right` are pressed as
named; a model turn is a **closed-loop turn by the lead enemy's bearing** (`Doom.turnBy(key, deg)` in
`site/games/doom/index.html`: hold the key until the angle has moved `deg`, max 1.5 s) — fixed holds cannot aim
into an 8° crosshair because the wasm build runs ~18 tics/s headless, presses under ~150 ms are dropped and a
180 ms press is ~20°; closed-loop lands within 3° (5→5.3, 10→12.3, 20→22.8, 45→47.5, 90→93.2). `retreat` →
move back (arena: turn right if blocked; real DOOM: turn right 90° when the last retreat did not move — a run died
in nukage backing into a wall for 80 ticks). `explore` → arena: forward, turn right at a wall; real DOOM: the
scripted **route** (below), then patrol.

**Probe, 50 situations (30 v3 + 20 new), gold = literal rules:**

| variant | acc | p_null | per-gold |
|---|---|---|---|
| v3 D5 intents (reference) | 1.000 (n=30) | .00 | retreat/engage/explore |
| v3 D2/D4 fine labels, object-location bearing sentences (reference) | .40–.80 | | turns 0/12 |
| A1 retreat first, one-condition rules, turn guardrail (turns removed with no enemy) | 1.000 | .13 | retreat 9/9, shoot 6/6, turn left 13/13, turn right 16/16, explore 6/6 |
| A2 orchestrator order (shoot, turns, retreat, default explore) + `shoot: … and ammo > 0` | .960 | .16 | explore 6/8 (→ retreat ×2) |
| **A3 = A1 without the turn guardrail** | **1.000** | **.02** | same as A1; guardrail-only random .20 |
| A4 = A1 with `The enemy is in the crosshair.` | 1.000 | .13 | same |
| A5 = A1 state, no rules (`Which action should the player take? Goal: survive…`) | .720 | .07 | retreat 1/9, explore 0/6 (→ shoot) |
| A6 shoot first, one-condition rules | 1.000 | .15 | (no retreat golds under that order) |
| A7 = A3 with `explore: applies when none of the above rules fire` | .840 | .02 | explore 0/8 (→ shoot 4, retreat 4) |
| A3 without the `The player has {h} health and {a} ammo.` sentence (FINAL grammar), n=48 | 1.000 | .00 | 7/7, 6/6, 13/13, 16/16, 6/6 (the 2 zero-ammo residual cases → turn left) |
| FINAL, hurt sentence moved after the enemy sentence | 1.000 | .00 | order-robust |
| FINAL state, no rules | .729 | .06 | retreat 0/7, explore 0/6 |
| FINAL robustness, 20 real-engine sentences (types, counts, walls, distances 3–30 cells) | 1.000 | .00 | 3/3, 5/5, 4/4, 6/6, 2/2 |
| mid-health crosshair (41/61/41/55/35/45 health, imp 5–20 cells) **with** the numbers sentence | .500 | .00 | retreat at 41 and 61 |
| same **without** the numbers sentence | 1.000 | .00 | shoot .88–.90 |
| enemy-mention sentences that are not a rule condition (`must move forward to reach the enemy`, `enemy ahead, off the crosshair`) with a default explore | 0/4 | | → shoot |

Chance .20.

**Arena engine changes for the aiming design:** cone shot (`_shootTarget` = the crosshair enemy), `SHOOT_RANGE =
AWARE_RANGE = 8`, line-of-sight filter (`_clearLine`), enemies wander (seeded) beyond 4 cells and chase inside it
(so sightings need turns — with chasing at 8 cells every enemy walked into the cone and the run had 1 turn), 8
enemies instead of 4. Literal policy, 240 ticks, seeds 7/1/2/3/4: 8 kills each, health 100/90/100/100/100, 5–11
turns.

**Recorded runs:** ASCII arena (`data/replays/doom.json`, seed 7, 240 ticks): 240/240 match the rules (explore
227, shoot 8, turn right 5), 8 shots / 8 in the crosshair / 8 kills, health 100, p_null .00, 122 ms. Real DOOM
(`data/replays/realdoom.json`, 200 decisions): **200/200 match the rules** — explore 133, shoot 64, turn 3
(2 right, 1 left), retreat 0; **4 kills**, health 86; the route reached the corridor door at decision 18 (`use`),
the two alcove zombiemen were in the crosshair at 11 cells at decision 20 and dead by 60, the third zombieman
was turned to and shot at decisions 73–77, a fourth at 94–111 in the nook; then the patrol walked the pillar
room ↔ nook loop with nothing in sight. Earlier recordings on the way: 199 explore / 0 kills (door `use` pressed
64 units out, out of use range); 134/200 / 3 kills (the health-number retreats); 200/200 / 3 kills / 113 shots
(imp at 18–20 cells, ammo gone); 200/200 / 5 kills / health 0 (free explorer walked into the nukage courtyard,
retreat blocked by a wall).

**E1M1 route (`ROUTE` in realdoom-logic.js, map units; spawn (1056,−3616) faces north; the start room's
east "exits" are windows, it only opens north; the corridor door at x=1536 is a DR door — one `use` press within
64 units opens it, a second press while it opens closes it again; HMP zombiemen at (2272,−2432), (2272,−2352),
(2912,−2816)):** (1230,−3000) → (1300,−2650) → (1480,−2450) → (1620,−2448, door) → (1950,−2440) → (1950,−2640)
→ (2380,−2640) → (2500,−2600) → (2700,−2600) → (2800,−2700) → (2850,−2830) → (2950,−2800); reached within
80 units each (`/tmp/probe4/route3.mjs`, 55 walking ticks); then patrol back to ROUTE[4] and forward. The
navigator turns (closed-loop) when the waypoint bearing exceeds 12°, else walks 550 ms; skips a non-door
waypoint after 8 ticks without moving; presses `use` at a door waypoint only once the walk has stopped against
it and not again for 8 s.

**Failure modes:** (1) any enemy sentence that is not one of the three conditions → `shoot`; (2) raw numbers in
the state (health) shift `shoot` → `retreat` at mid health; (3) the default rule must be an explicit condition;
(4) with a crosshair enemy and no ammo the literal list has no answer (guardrail null: 2 probe cases → `turn
left`); (5) the arena's crosshair is a 45° cone — the 8-heading geometry, not the model, sets that; (6) real DOOM
is navigation-limited beyond the route (a free explorer finds the nukage).

**Honest caption (on the plate):** "Each tick the model reads the situations that apply, pre-computed as
agent-subject sentences ('the player is badly hurt', 'the player has a zombieman in the crosshair, 6 cells
ahead', 'the player must turn left to face the imp') and picks retreat, shoot, turn left, turn right or explore
from five one-condition rules; the engine only presses the key (a closed-loop turn by the enemy's bearing) and
walks a scripted route from the E1M1 spawn to the first zombiemen. It matched the rule list 48/48 in the probe
(chance .20; .73 with no rules) and 200/200 in the recorded E1M1 run: 3 turns, 64 shots, 4 kills, health 86. The
v3 form 'the enemy is 5 cells to the left' never fired a turn (0/12); the rewrite to 'the player must turn left
to face the enemy' did. Raw health numbers had to go: '41 health' with an enemy in the crosshair read as danger
(retreat .56 over shoot)."

---

## Before / after

| | before (v2/v3) | after (v4) |
|---|---|---|
| Snake probe | v2 grammar .774 on the 62 boards (food-ward .72 vs .53 chance in the v2 probe) | .984 (n=62, chance .48, guardrail-only random .48); rules-in-question variants .82–.95 with p_null ~.9 |
| Snake recorded run (seed 7, 300 ticks) | 2 food, food-ward .55 vs .39 random-safe, no gold | 24 food, food-ward .926, 216/217 rule agreement, trapped at tick 217 (length 27), p_null .06 |
| Doom aiming probe | fine labels .40–.80, turns 0/12 over five phrasings; intents 30/30 | 48/48 (n=50 incl. 2 residuals; turns 29/29; chance .20; no rules .73); 20/20 real-engine sentences |
| Doom arena run | 62/62 intents, 2 shots / 2 kills, dies tick 63 (engine aims) | 240/240 over 5 labels, 5 model turns, 8 shots / 8 in crosshair / 8 kills, health 100 (model aims) |
| Real DOOM run (200 decisions) | 200/200 intents, 199 explore / 1 engage, 0 shots, 0 kills (never left the start rooms) | 200/200 over 5 labels, 133 explore / 64 shoot / 3 turn, 4 kills, health 86 (scripted route + patrol; model aims) |
