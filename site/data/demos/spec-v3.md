# Typical site — demo spec v3: Drive and Doom (2026-09-22)

Supersedes `demo-spec-v2.md` §4–5. Everything below is `[probe]` against the live server
(`POST http://localhost:8787/api/decide`, `OzLabs/typical-small`, MPS, M4 Pro), scripts in `/tmp/probe3/*.py`
(`lib.py` = HTTP twin of the old in-process lib; `drive_sits.py` reproduces the 30 v2 Drive situations bit-exact
from the same seed and adds 20 lane-change situations; `doom_sits.py` parses the 30 v2 Doom strings). The engines
in `site/js/games/{drive,doom,realdoom-logic}.js` implement the chosen grammars verbatim and their self-tests
(`node <file>`) pin the exact sentences; `scripts/record_games.mjs` / `record_realdoom.mjs` now record per-frame
`gold` (the rule list applied literally) and report `rule_agreement`.

## What JevPilot does, and what of it transfers

`src/jev-request.js` (standardagents/jevpilot): a JSON state with `driving_style` sentences that are included
**only when relevant** (traffic / intersection clauses appear only if there is traffic / an intersection), compact
tables whose shared column values are factored out once, pre-computed candidate rows (each sampled steering+speed
vector carries `progress`, `lane_error`, `conflicts`, `stop_at_line`), and **two factored questions**: `motion`
(drive/stop) and `vector` (which path, "assuming drive"). Candidates with a single option are fixed client-side and
never sent. What transfers to a 0.6B text head trained on DecisionMix v2 L4 precedence lists: (1) state = only the
situations that apply, (2) each situation sentence pre-computes the comparison the rule needs ("closing on a slower
vehicle", "must take the exit") so the rule is one condition, (3) the candidate set is the legality guardrail, (4)
the engine executes; the model decides. Tables and per-candidate numeric rows do not transfer (no arithmetic).

## Rules learned about the head (these drove every choice)

- **One condition per rule.** Any rule with `or` fires everywhere: Drive `change lane right: … exit …, or …
  slower vehicle …` → .30 (C) / .61 (FINAL+OR) vs 1.00 without; v2's compound list .34 on the same 50.
- **Rule position ≤ 4 before the default is safe; position 6 is dead.** The accelerate rule at position 6
  fired 0/5 under four phrasings; at position 4 the same rule fired 4–5/5. Training L4 lists are 3 rules + default.
- **Precedence is soft.** The model does not do strict first-match: with `accelerate` (pos 4, condition true)
  listed before `change lane left` (pos 5), it still answered change lane left 13/13. So the *literal* rule list
  must equal the intended policy — make the situation sentences mutually exclusive where the rules overlap.
- **Agent-subject situation sentences match; object-location sentences do not.** `The car must take the exit on
  the right in 200 m.` 11/11; `The exit on the right is 200 m ahead.` 0/11; `The destination exit is 200 m ahead on
  the right.` 0/11. Doom: `The nearest enemy is 5 cells to the left.` never fires `turn left` (0/12, five phrasings).
- **Negated conditions are weak.** `applies when no vehicle is ahead within 50 m` 0/5; `applies when the road
  ahead is clear for 50 m` 5/5 at the same position.
- **Sentence order matters** (fixed order 70/70; two shuffles .886 / .929 — the loss is always exit-vs-accelerate).
- **Criteria for labels the guardrail removed → null mass**, exactly the trained `null_group` behaviour: gating
  `accelerate` at the limit gave mean p_null .49 on the recorded run (.68 on cruise ticks) with the argmax still
  right; offering it always (no-op at the limit) gives .22. Keep criteria == the full label set; keep guardrails to
  real illegality.
- **Label priors are strong:** `shoot` takes any enemy mention; the first-listed label takes the residual when the
  model thinks nothing fires (Doom `move back` 23/24 after removing shoot).

---

## 4. Drive

**Chosen: V6 — situations-only state, 7 one-condition rules in precedence order, legality guardrail.**

**State template (exact, fixed order; only sentences whose situation holds):**
```
The car is in lane {l} of 3 at {v} km/h, {below|at} the speed limit of 60 km/h.
[A pedestrian is crossing the car's lane {d} m ahead.]              pedestrian in own lane ≤ 20 m ahead
[The traffic light {d} m ahead is red.]                              red light ≤ 60 m ahead
[The car is closing on a slower vehicle {d} m ahead.                 own-lane lead ≤ 20 m and slower
 | The lead vehicle is {d} m ahead at {lv} km/h.]                    else own-lane lead ≤ 50 m (actual speed)
[The car must take the exit on the right in {d} m.]                  exit ≤ 300 m and lane < 3
[The road ahead is clear for 50 m.]                                  no own-lane vehicle ≤ 50 m ahead
[The lane to the left is clear.] [The lane to the right is clear.]   lane exists, no vehicle within 15 m
```
Example: `The car is in lane 2 of 3 at 45 km/h, below the speed limit of 60 km/h. The traffic light 60 m ahead is red. The car is closing on a slower vehicle 18 m ahead. The lane to the left is clear.`

**Question (exact; `QUESTION` in drive.js, rendered like `query_text`):**
```
Which manoeuvre applies for driving the car? Apply the rules in the stated precedence order; the first matching rule wins.
stop: applies when a pedestrian is crossing the car's lane  brake: applies when the traffic light ahead is red  change lane right: applies when the car must take the exit on the right  accelerate: applies when the road ahead is clear for 50 m  change lane left: applies when the car is closing on a slower vehicle ahead  follow: applies when the car is closing on a slower vehicle ahead  hold speed: applies when none of the above rules fire
```
Criteria dict (= `RULES`, order is precedence and label order):
`{stop, brake, change lane right, accelerate, change lane left, follow, hold speed}` as above. Labels = the
offered subset of these, same order. Overtaking is left-only by design (the right-lane rule is the exit; adding
"or … slower vehicle" to it breaks everything, see above); `follow` = match the lead vehicle's speed.

**Candidate policy (`Drive.candidates()`):** all 7, minus `change lane right` unless lane < 3 and that lane is
clear (no vehicle within 15 m), minus `change lane left` unless lane > 1 and clear and the exit is not within
300 m (no lane change away from the exit). `stop`/`brake`/`follow`/`accelerate`/`hold speed` are always offered
(no-ops at 0 km/h, at the limit, or with no lead). Scripted policy `greedyPolicy` = the rule list applied literally
over the offered labels = the gold.

**Probe, 70 situations (30 v2 + 20 lane-change + 20 corner: red light with a clear lane / below the limit /
while closing, exit while closing, lead at 30–45 m same speed, v = 0 at a red light), gold = literal rules:**

| variant | acc | p_null | ms | per-gold |
|---|---|---|---|---|
| V1 current (v2 state + v2 compound question, 6 labels) | .340 (n=50) | .00 | 160 | stop 5/5, brake 5/5, accelerate 5/5, lane left 0/10, lane right 1/20 (→ brake), hold 1/5 |
| V2 JevPilot: situations-only + compound-with-clear-clause rules, 6 labels | .520 | .00 | 126 | lane right 20/20, stop 5/5, everything else → change lane right |
| V3 all predicates as true/false sentences + same rules | .520 | .00 | 134 | identical to V2 — negated facts buy nothing |
| V4 guardrail + single-condition rules where the guardrail allows, 6 labels | .520 | .11 | 116 | stop/lane right ok; brake, lane left → stop; guardrail-only random .20 |
| V5 7 labels (split `follow` off `brake`), S_sit, criteria all 7, accelerate at pos 6 | .680 | .14 | 117 | lane left 13/13, follow 10/10, exit 0/11, accelerate 0/5 |
| V5b same, criteria = offered only | .480 | .00 | 118 | lane left 3/13 (→ follow) — keep criteria == full set |
| V5c 7 labels with `or` on change-lane-right (overtake either side) | .300 | .00 | 117 | disjunction kills it |
| **V6 FINAL** (agent-subject exit sentence, `road ahead is clear` at pos 4, fixed order) | **1.000 (n=70)** | .23 | 115 | stop 5/5, brake 11/11, lane right 14/14, accelerate 10/10, lane left 15/15, follow 11/11, hold 4/4 |
| V6 sentence order shuffled (2 seeds) | .886 / .929 | .27 | — | exit → accelerate 6/14, 4/14 |
| V6 no guardrail (all 7 always offered) | .771 | .00 | 148 | follow → change lane left 11/11 (lane not clear), hold → accelerate at the limit 4/8 |
| V6 no rules (`Which manoeuvre is safe and makes progress…`) | .157 | .05 | 98 | hold speed ×59 |
| V6 state + v2 compound question | .386 | .01 | 178 | lane left 0/15, follow 0/11 |
| guardrail-only random baseline (V6 candidate sets) | .160 | | | |

Chance among offered .16. Every number is the argmax over offered labels; p_null is reported, never used.

**Recorded run (`data/replays/drive.json`, seed 7, 232 ticks):** 231/231 decisions match the rule list; 18/18
red lights respected, 0 run; 3 lane changes (one overtake left at tick 24, two right toward the exit at 194–195);
2 `follow` ticks; arrived; 0 collisions; mean p_null .22 (cruise ticks where only the default applies sit at
.6–.7, the trained "nothing fires" signal); 113 ms/decision. Engine changes for the recording: road 1600 m, exit
at 1500 m, 16 cars bunched in the first 660 m at lane-dependent 15–50 km/h, traffic waits behind anything within
10 m and at red lights (a stopped ego was being rammed), green 20 ticks; render-drive.js imports
`ROAD_LENGTH`/`DEST_POS` from the engine.

**Failure modes:** (1) any `or` in a rule; (2) exit-vs-accelerate flips if the clear-road sentence precedes the
exit sentence; (3) `follow` ticks carry p_null ≈ .57 because the change-lane-left rule fires but its label is not
offered; (4) a slower car in lane 1 is followed, not passed on the right (policy, not model); (5) 7 labels is at
the edge — position 6 works only because `follow` shares its condition with position 5.

**Honest caption (used on the tile):** "Each tick the road is rendered as the situations that apply, seven
one-condition rules are in the prompt in precedence order, and the model picks one manoeuvre from the legal set.
It matched the rule list on 70/70 scripted situations (chance .16; .16 random-among-legal; .16 with no rules) and
on every tick of the recorded run: 18/18 red lights, 3 lane changes, exit taken. It cannot take a rule with an
'or' in it, and it reads sentence order."

---

## 5. Doom (ASCII arena `doom.js`; real DOOM `realdoom-logic.js`)

**Chosen: D5 — coarse intents `retreat / engage / explore`, two situation facts, the engine aims.**

**State template (exact, fixed order):**
```
The player has {h} health and {a} ammo.
[The player is badly hurt.]                                             health < 30
{The player has an enemy in sight, {d} cells {ahead in the crosshair|to the left|to the right|behind}.
 | The player has {n} enemies in sight; the nearest is {d} cells {…}.   (crosshair enemy listed first)
 | The player sees no enemy.}
[A wall is ahead.]
```
Real DOOM adds the monster type: `The player has an enemy in sight: an imp 5 cells ahead in the crosshair.` /
`The player has 2 enemies in sight; the nearest is a zombieman 8 cells to the left.` "In sight" in real DOOM =
`visible` (new `P_CheckSight` field in the wasm state, `vendor/doom/typical.patch`, rebuilt 2026-09-22) and not
behind (|bearing| ≤ 100°) and ≤ 40 cells; before this the state listed the six nearest monsters through walls,
which is why the old run had 90 shots and 0 targets.

**Question (exact; `QUESTION` in doom.js, shared by realdoom):**
```
Which action applies for the player? Apply the rules in the stated precedence order; the first matching rule wins.
retreat: applies when the player is badly hurt  engage: applies when the player has an enemy in sight  explore: applies when the player sees no enemy
```
Criteria dict = `RULES = {retreat, engage, explore}`; labels = all three, always.

**Candidate policy:** the three intents are always offered. `resolve()`/`resolveIntent()` (engine, stated on the
page): retreat → move back (turn right if blocked); engage → shoot if the leading enemy is in the crosshair (≤ 6
cells in the arena; ammo > 0), else turn toward it (behind → turn right), else move forward; explore → move
forward, turn right when a wall is ahead. Human keys still send raw actions; they pass through.

**Probe, the 30 v2 situations (6 per gold), gold = literal rules:**

| variant | acc | p_null | ms | per-gold |
|---|---|---|---|---|
| D1 current (v2 state + v2 5-rule question) | .200 | .00 | 101 | shoot 29/30 |
| D2 JevPilot: situations-only agent-subject state, 5 one-condition rules (`move back: badly hurt`, `shoot: has an enemy in the crosshair`, `turn left: nearest enemy is to the player's left`, …) | .400 | .00 | 91 | move back 6/6, shoot 6/6, turns 0/12 (→ shoot), forward 0/6 (→ shoot) |
| D2 + scene distractors (walls/doors) | .400 | .00 | 98 | same |
| D3 = D2 + crosshair guardrail on shoot | .400 | .49 | 88 | turns 0/12 → move back (first label); guardrail-only random .23 |
| D4 4 labels, `turn toward the enemy` (engine picks the side), 3 rules + default; 4 crosshair/off-axis phrasings (aim/spot, lined-up/off-to-the-side, target/flank, …) and label swaps (fire/aim, shoot/take aim, shoot/turn to face) | .40–.80 | .00 | 80–90 | whichever of {shoot, turn} is listed at position 2 takes all 18 enemy cases; the head does not separate "in the crosshair" from "to the left" |
| **D5 FINAL** coarse `retreat / engage / explore`, explicit explore rule | **1.000** | .00 | 77 | 6/6, 18/18, 6/6 |
| D5 no rules (`Which action should the player take? Goal: survive…`) | .600 | .04 | 64 | engage ×30 |
| D6 fine labels gated by the crosshair (`shoot` xor `turn toward the enemy`), criteria = offered | 1.000 | .00 | 77 | same decision as D5 wearing key names — rejected as less honest; guardrail-only random .33 |
| D5 robustness, 24 real-DOOM-style sentences (types, 2–3 enemies, walls, health 29/30) | .958 | .00 | 80 | one `engage` → `retreat` at 45 health / 0 ammo / demon 3 cells away |

Chance .33 (.20 for the 5-label variants).

**Recorded runs:** ASCII arena (`data/replays/doom.json`, seed 7): 62/62 decisions match the rules (explore 39,
engage 21, retreat 2 → 48 forward / 11 turn right / 2 shoot / 1 back), 2 kills, dies at tick 63 (the 8-heading
aiming is the engine's weakness, not the model's), p_null .00, 78 ms. Real DOOM (`data/replays/realdoom.json`,
200 decisions): 200/200 match the rules — 199 `explore`, 1 `engage` (an imp glimpsed at 36 cells, tick 122), 0
shots, 0 kills, health 106. The E1M1 start rooms are >20 cells from any monster and the forward/turn-right explorer
(also random-turn, wall-alternating, seek-the-nearest and a wake-up shot: all tried, 0–1 sightings in 250 ticks)
never leaves them. That is an engine/navigation limit and is stated on the tile; the fallback arena shows engagements.

**Failure modes:** (1) the model cannot aim — bearing words never map to `turn left/right`, and `shoot` claims
any enemy; (2) `0 ammo` reads as danger (engage → retreat .54); (3) the real-DOOM run is almost all `explore`.

**Honest caption (used on the tile):** "Each tick the model reads two facts (badly hurt? an enemy in line of
sight?) and picks an intent — retreat, engage, explore — and the engine aims and presses the key. That split is the
finding: it matches the three rules 30/30 (chance .33; .60 with no rules), but when it was asked to aim itself
(shoot / turn left / turn right) it said 'shoot' for any enemy mention, 0/12 turns over five phrasings. In the
recorded E1M1 run its explorer never left the first rooms, so it decided 'explore' 199 times and 'engage' once."

---

## Before / after

| | before (v2) | after (v3) |
|---|---|---|
| Drive probe | .533 (n=30), lane changes 0/10 | 1.000 (n=70), lane changes 29/29; no-guardrail .771; guardrail-only .16; no-rules .16 |
| Drive recorded run | 10 m in 240 ticks, p_null .66 | 1508 m, arrived, 18/18 reds, 3 lane changes, 231/231 rule agreement, p_null .22 |
| Doom probe | .200 (n=30), shoot 29/30 | 1.000 (n=30) on intents; aiming variants .40–.80 (documented failure) |
| Doom arena run | 240 ticks, 0 kills, 12 shots / 0 in crosshair, p_null .78 | 62/62 rule agreement, 2 shots / 2 in crosshair, 2 kills, dies tick 63 |
| Real DOOM run | 90 shots, 0 targets (monsters listed through walls) | 200/200 rule agreement, 0 shots, 1 sighting (LOS-aware state) |
