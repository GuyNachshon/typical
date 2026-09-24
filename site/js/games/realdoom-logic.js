// Pure grammar/policy helpers for real DOOM (js/games/render-realdoom.js and
// scripts/record_realdoom.mjs both import this - no DOM, safe in Node and the browser). Kept
// separate from render-realdoom.js so the recorder can drive the exact same sentence grammar
// and legal-move rules the live card uses, without dragging THREE.js/loop.js into Node.
// Same demo-spec v4 design as js/games/doom.js: the state pre-computes the turn side ("The player
// must turn left to face the imp."), the model picks retreat / shoot / turn left / turn right /
// explore, and the engine only presses the key (a closed-loop turn by the lead monster's bearing).
import { LABELS, HURT_BELOW } from './doom.js';

const CELL_UNITS = 64; // map units per Doom grid cell (doom-wasm-notes.md)
const CROSSHAIR_DEG = 8;
const BEHIND_DEG = 100; // |bearing| beyond this = behind the player = not "in sight"
const SIGHT_CELLS = 12; // Doom.state() lists the nearest monsters map-wide; "in sight" = line of sight (visible), not behind, within pistol range (an imp 18 cells off across the nukage ate 69 shots and all the ammo in a recorded run)

export const ACTIONS = ['move forward', 'move back', 'turn left', 'turn right', 'shoot', 'use'];

// Chocolate Doom key id + hold duration (ms) for each key press - press(key, ms) on
// window.Doom (site/games/doom/index.html's harness).
export const KEY_FOR_MOVE = {
  // 'move forward' held longer than the others: E1M1's spawn corridor + pillar room alone are
  // >1700 units from the nearest zombieman, and each decision is one key press - a short hold
  // means most of a 200-decision recorded run is still walking there. 550ms covers noticeably
  // more ground per decision without outrunning the 400ms tick interval by much (loop.js's
  // ticker just skips a tick while the press is still held, same as it already does at 350ms).
  'move forward': ['forward', 550],
  'move back': ['back', 350],
  'turn left': ['left', 180], // keyPress() replaces these with a closed-loop turn by degrees
  'turn right': ['right', 180],
  shoot: ['fire', 120],
  use: ['use', 150], // explore only: opens the corridor door on the ROUTE
};

export function cellsOf(units) {
  return Math.max(0, Math.round(units / CELL_UNITS));
}

const inCrosshair = (m) => m.in_crosshair || Math.abs(m.bearing ?? 0) <= CROSSHAIR_DEG;

// Monsters "in sight": line of sight (state.visible, P_CheckSight in the wasm patch), not behind,
// within SIGHT_CELLS; crosshair first, then nearest first.
export function inSight(state) {
  return (state.monsters ?? [])
    .filter((m) => m.visible !== false && cellsOf(m.dist) <= SIGHT_CELLS && Math.abs(m.bearing ?? 0) <= BEHIND_DEG)
    .sort((a, b) => {
      const ca = a.in_crosshair ? 0 : 1;
      const cb = b.in_crosshair ? 0 : 1;
      return ca === cb ? a.dist - b.dist : ca - cb;
    });
}

// The model's candidates are the legality guardrail: `shoot` is only offered when there is
// something to shoot at and ammo to do it with. Offering it with nothing in sight meant the model
// occasionally picked it and the engine fired into a wall while the state sentence — correctly —
// said the player saw no enemy.
export function candidatesFor(state) {
  if (state.in_level === false) return [];
  const canShoot = (state.ammo ?? 0) > 0 && inSight(state).length > 0;
  return LABELS.filter((l) => l !== 'shoot' || canShoot);
}

const replayKey = (sentence, candidates) => JSON.stringify([sentence, candidates]);

// A recorded model call is reusable whenever the rendered state sentence and candidate list are
// identical: those are the model's complete inputs. The live game can diverge from the original
// wall-clock recording while still showing the exact model output for every state it revisits.
export function replayIndex(doc) {
  const index = new Map();
  for (const d of doc?.decisions ?? []) {
    if (!d?.desc || !Array.isArray(d.candidates) || !Array.isArray(d.probs) || !d.candidates.includes(d.move)) continue;
    index.set(replayKey(d.desc, d.candidates), {
      move: d.move,
      probs: Object.fromEntries(d.candidates.map((label, i) => [label, d.probs[i] ?? 0])),
      p_null: 0,
      ms: d.ms ?? 0,
    });
  }
  return index;
}

export function recordedDecision(index, sentence, candidates) {
  return index?.get(replayKey(sentence, candidates)) ?? null;
}

// state -> the sentence the model reads: only the situations that apply, worded exactly as the
// rule conditions (js/games/doom.js RULES), monster type included, in a fixed order. No
// health/ammo numbers (see doom.js describe()). The lead monster (crosshair first, then
// nearest) is the one the sentence is about.
export function describeDoom(state) {
  const sentences = [];
  if (state.health < HURT_BELOW) sentences.push('The player is badly hurt.');
  const seen = inSight(state);
  const lead = seen[0];
  const typed = (m) => `${/^[aeiou]/i.test(m.type) ? 'an' : 'a'} ${m.type}`;
  if (seen.length > 1) sentences.push(`The player has ${seen.length} enemies in sight.`);
  if (!lead) sentences.push('The player sees no enemy.');
  else if (inCrosshair(lead)) {
    const d = cellsOf(lead.dist);
    sentences.push(`The player has ${typed(lead)} in the crosshair, ${d} cell${d === 1 ? '' : 's'} ahead.`);
  } else sentences.push(`The player must turn ${lead.bearing < 0 ? 'left' : 'right'} to face ${seen.length > 1 ? 'the nearest enemy' : `the ${lead.type}`}.`);
  if (state.blocked_ahead) sentences.push('A wall is ahead.');
  return sentences.join(' ');
}

// explore navigator: a scripted route from the E1M1 spawn to the first zombiemen, then a
// minimal wall-turn fallback. Map units, Doom frame (x east, y north, angle CCW from east).
// The spawn (1056,-3616) room only opens north; its east "exits" are windows. The corridor
// door at x=1536 is a DR door: one `use` press opens it (a second press while it is opening
// closes it again), then walking into it until it is open. The two HMP zombiemen stand in
// the raised alcove at (2272,-2432)/(2272,-2352) and come to the player once seen.
export const ROUTE = [
  { x: 1290, y: -3000 }, // north along the start room's east side, in line with the corridor mouth (x=1230 snagged the corner for ~15 s of film)
  { x: 1300, y: -2650 }, // the corridor north-east
  { x: 1480, y: -2450 }, // corridor end, facing the door
  { x: 1620, y: -2448, door: true }, // past the door: blocked here -> use, then walk in
  { x: 1950, y: -2440 }, // pillar room, in front of the alcove
  { x: 1950, y: -2640 }, // round the alcove's south side (its wall is y=-2544)
  { x: 2380, y: -2640 },
  { x: 2500, y: -2600 }, // the room's east passage
  { x: 2700, y: -2600 },
  { x: 2800, y: -2700 }, // south-east into the third zombieman's nook at (2912,-2816)
  { x: 2850, y: -2830 },
  { x: 2950, y: -2800 },
];
const PATROL_FROM = 4; // once the route is done, patrol between ROUTE[4] (pillar room) and its end
const WP_RADIUS = 80;
const TURN_TOL_DEG = 12;
const STUCK_TICKS = 4; // explore ticks (~1 s each, live) without moving 8 units -> skip the waypoint
const DOOR_RETRY_MS = 8000; // a DR door takes ~4 s to open at the wasm build's tic rate
const DEFAULT_TURN_DEG = 60;
let wpIndex = 0;
let wpDir = 1; // +1 outbound, -1 walking the route back (patrol)
let wpStuck = 0;
let lastPos = null;
let usedAt = -Infinity;
let navTurnDeg = DEFAULT_TURN_DEG; // what the last explore turn asked for (keyPress reads it)
let retreatPos = null; // where the last retreat started; no movement since -> the way back is blocked

export function resetNav() {
  wpIndex = 0;
  wpDir = 1;
  wpStuck = 0;
  retreatPos = null;
  lastPos = null;
  usedAt = -Infinity;
  navTurnDeg = DEFAULT_TURN_DEG;
}

const norm = (d) => ((((d + 180) % 360) + 360) % 360) - 180;

function exploreAction(state, now = Date.now()) {
  // Past the last waypoint the route is walked back to PATROL_FROM and forward again: a free
  // explorer wandered off the far end into the nukage courtyard and died there (health 0 at
  // decision 200 of a recorded run). Monsters woken along the way come to the patrol.
  if (wpIndex >= ROUTE.length) {
    wpDir = -1;
    wpIndex = ROUTE.length - 2;
  } else if (wpDir < 0 && wpIndex < PATROL_FROM) {
    wpDir = 1;
    wpIndex = PATROL_FROM + 1;
  }
  const wp = ROUTE[wpIndex];
  if (Math.hypot(wp.x - state.x, wp.y - state.y) < WP_RADIUS) {
    wpIndex += wpDir;
    return exploreAction(state, now);
  }
  if (lastPos && Math.hypot(state.x - lastPos.x, state.y - lastPos.y) < 8) wpStuck += 1;
  else wpStuck = 0;
  lastPos = { x: state.x, y: state.y };
  if (wpStuck > STUCK_TICKS && !wp.door) { // a door waypoint is never skipped: it retries `use`
    wpIndex += wpDir;
    wpStuck = 0;
    return exploreAction(state, now);
  }
  // bearing to the waypoint, positive = to the right (same convention as state.monsters)
  const bearing = -norm((Math.atan2(wp.y - state.y, wp.x - state.x) * 180) / Math.PI - state.angle);
  if (Math.abs(bearing) > TURN_TOL_DEG) {
    navTurnDeg = Math.abs(bearing);
    return bearing > 0 ? 'turn right' : 'turn left';
  }
  // `use` reaches 64 units; blocked_ahead fires 64 units out - so press it only once the walk
  // has actually stopped against the door (wpStuck >= 1), and never while it is still opening.
  if (state.blocked_ahead && wp.door && wpStuck >= 1 && now - usedAt > DOOR_RETRY_MS) {
    usedAt = now;
    wpStuck = 0;
    return 'use';
  }
  return 'move forward';
}

// Label -> key action. The model aims (shoot / turn left / turn right are its own labels); the
// engine walks for retreat/explore. Raw actions pass through for the human override.
// The model decides *what* to do; the engine carries it out, and "engage that enemy" is not one
// key press. A pistol shot at 9 cells is a pixel-wide sprite and a coin-flip hitscan, which is
// what "it shoots without aiming" looks like from the outside. So a `shoot` decision first
// centres the target (a closed-loop turn by its bearing, to 3 degrees rather than the 8 the
// crosshair test allows), then closes to inside FIRE_CELLS, and only then fires. Same split as
// retreat -> move back and explore -> navigate: the label is the decision, not the keystroke.
const FINE_AIM_DEG = 3;
const FIRE_CELLS = 6;

export function resolveIntent(state, move) {
  if (move === 'shoot') {
    const lead = inSight(state)[0];
    if (lead) {
      const off = lead.bearing ?? 0;
      if (Math.abs(off) > FINE_AIM_DEG) {
        navTurnDeg = Math.abs(off);
        return off > 0 ? 'turn right' : 'turn left';
      }
      if (cellsOf(lead.dist) > FIRE_CELLS && !state.blocked_ahead) return 'move forward';
    }
    return 'shoot';
  }
  if (move === 'retreat') {
    // move back, unless the last retreat did not move the player (a wall behind: a recorded run
    // died in nukage backing into a wall 80 ticks running) - then turn to find another way out.
    const stuck = retreatPos && Math.hypot(state.x - retreatPos.x, state.y - retreatPos.y) < 8;
    retreatPos = { x: state.x, y: state.y };
    if (stuck) {
      navTurnDeg = 90;
      return 'turn right';
    }
    return 'move back';
  }
  retreatPos = null;
  if (move === 'explore') return exploreAction(state);
  return move;
}

// Key press for an action: [key, holdMs] or [key, { deg }] for a closed-loop turn (games/doom/
// index.html Doom.turnBy). A model turn turns by the lead monster's bearing so the next state
// has it in the crosshair; an explore turn follows the navigator's request.
export function keyPress(state, move, action) {
  if (action !== 'turn left' && action !== 'turn right') return KEY_FOR_MOVE[action];
  const key = action === 'turn left' ? 'left' : 'right';
  if (move === 'explore' || move === 'retreat' || move === 'shoot') return [key, { deg: navTurnDeg }];
  const lead = inSight(state)[0];
  return [key, { deg: lead ? Math.min(120, Math.max(8, Math.abs(lead.bearing ?? 0))) : DEFAULT_TURN_DEG }];
}

// Scripted policy = RULES applied literally (never calls the model) over the offered labels.
// With a crosshair target and no ammo nothing fires; the residual goes to the first label.
export function scriptedPolicy(state) {
  if (state.health < HURT_BELOW) return 'retreat';
  const lead = inSight(state)[0];
  if (!lead) return 'explore';
  if (inCrosshair(lead)) return (state.ammo ?? 0) > 0 ? 'shoot' : 'retreat';
  return lead.bearing < 0 ? 'turn left' : 'turn right';
}

function selfTest() {
  const s1 = { health: 84, ammo: 40, blocked_ahead: false, monsters: [{ type: 'zombieman', dist: 320, bearing: 3, in_crosshair: true }] };
  console.assert(describeDoom(s1) === 'The player has a zombieman in the crosshair, 5 cells ahead.', 'crosshair sentence');
  console.assert(candidatesFor(s1).join() === 'retreat,shoot,turn left,turn right,explore', 'with a target and ammo, all five labels are offered');
  console.assert(!candidatesFor({ ...s1, monsters: [] }).includes('shoot'), 'nothing in sight: shoot is not on the table');
  console.assert(!candidatesFor({ ...s1, monsters: [{ type: 'imp', dist: 400, bearing: 20, visible: false }] }).includes('shoot'), 'a monster behind a wall is not a target');
  const replay = replayIndex({ decisions: [{ desc: 'state', candidates: ['a', 'b'], probs: [0.2, 0.8], move: 'b', ms: 12 }] });
  console.assert(recordedDecision(replay, 'state', ['a', 'b'])?.probs.b === 0.8, 'recorded model output is keyed by its complete inputs');
  console.assert(recordedDecision(replay, 'state', ['b', 'a']) === null, 'candidate order is part of the recorded input');
  console.assert(scriptedPolicy(s1) === 'shoot' && resolveIntent(s1, 'shoot') === 'shoot' && keyPress(s1, 'shoot', 'shoot')[0] === 'fire', 'crosshair target, centred and close -> fire');
  const far = { ...s1, monsters: [{ type: 'zombieman', dist: 600, bearing: 1, in_crosshair: true }] };
  console.assert(resolveIntent(far, 'shoot') === 'move forward', 'a shoot decision 9 cells out closes the distance first');
  const wide = { ...s1, monsters: [{ type: 'zombieman', dist: 320, bearing: 6, in_crosshair: true }] };
  console.assert(resolveIntent(wide, 'shoot') === 'turn right' && keyPress(wide, 'shoot', 'turn right')[1].deg === 6, 'a shoot decision 6 degrees off centres the target first');

  const s2 = { health: 20, ammo: 0, blocked_ahead: true, monsters: [{ type: 'imp', dist: 700, bearing: -40, in_crosshair: false }, { type: 'demon', dist: 3000, bearing: 10, in_crosshair: false }, { type: 'zombieman', dist: 500, bearing: 2, in_crosshair: true, visible: false }] };
  console.assert(describeDoom(s2) === 'The player is badly hurt. The player must turn left to face the imp. A wall is ahead.', 'hurt + pre-computed turn side + far monster and wall-hidden monster dropped + wall');
  console.assert(!candidatesFor(s2).includes('shoot'), 'no shoot without ammo');
  console.assert(scriptedPolicy(s2) === 'retreat' && resolveIntent(s2, 'retreat') === 'move back', 'badly hurt -> retreat -> move back');
  console.assert(resolveIntent(s2, 'turn left') === 'turn left' && JSON.stringify(keyPress(s2, 'turn left', 'turn left')) === '["left",{"deg":40}]', 'a model turn is a closed-loop turn by the lead bearing');
  console.assert(resolveIntent(s2, 'turn right') === 'turn right', 'raw actions pass through');

  const s3 = { health: 100, ammo: 50, blocked_ahead: false, monsters: [{ type: 'imp', dist: 500, bearing: 160, in_crosshair: false }] };
  console.assert(describeDoom(s3) === 'The player sees no enemy.', 'a monster behind is not in sight');
  console.assert(scriptedPolicy(s3) === 'explore', 'scripted explores with nothing in sight');

  const s4 = { health: 60, ammo: 8, blocked_ahead: false, monsters: [{ type: 'imp', dist: 400, bearing: 30, in_crosshair: false }, { type: 'zombieman', dist: 600, bearing: -2, in_crosshair: true }] };
  console.assert(describeDoom(s4) === 'The player has 2 enemies in sight. The player has a zombieman in the crosshair, 9 cells ahead.', 'crosshair monster leads even if farther; count sentence first');
  s4.monsters[1].bearing = 20;
  s4.monsters[1].in_crosshair = false;
  console.assert(describeDoom(s4).endsWith('The player must turn right to face the nearest enemy.') && scriptedPolicy(s4) === 'turn right', 'two off-axis enemies -> turn toward the nearest');

  // route navigator: spawn faces north (90); first waypoint is north-north-east -> turn right by
  // the bearing, then walk; at the door waypoint a blocked player uses once, then keeps walking.
  resetNav();
  const spawn = { x: 1056, y: -3616, angle: 90, blocked_ahead: false, monsters: [], health: 100, ammo: 50 };
  console.assert(resolveIntent(spawn, 'explore') === 'turn right' && keyPress(spawn, 'explore', 'turn right')[1].deg > 10, 'explore turns toward the first waypoint');
  const aimed = { ...spawn, angle: 74 };
  console.assert(resolveIntent(aimed, 'explore') === 'move forward', 'explore walks once aimed');
  const atDoor = { ...spawn, x: 1520, y: -2448, angle: 0, blocked_ahead: true };
  resetNav();
  for (const wp of ROUTE.slice(0, 3)) resolveIntent({ ...atDoor, x: wp.x, y: wp.y, blocked_ahead: false }, 'explore'); // stand on each waypoint -> reached
  const t0 = 1000;
  const acts = [0, 400, 800, 1200].map((dt) => exploreAction(atDoor, t0 + dt));
  console.assert(acts.join() === 'move forward,use,move forward,move forward', `door: walk up, use once, then walk in (${acts})`);
  resetNav();
  // patrol: past the route's end the index walks back toward PATROL_FROM
  for (const wp of ROUTE) resolveIntent({ ...spawn, x: wp.x, y: wp.y }, 'explore');
  resolveIntent({ ...spawn, x: ROUTE[ROUTE.length - 1].x, y: ROUTE[ROUTE.length - 1].y }, 'explore');
  console.assert(wpDir === -1 && wpIndex === ROUTE.length - 2, `patrol turns back at the end (${wpDir}, ${wpIndex})`);
  // retreat: move back, but a second retreat from the same spot turns
  resetNav();
  const hurt = { ...spawn, health: 20 };
  console.assert(resolveIntent(hurt, 'retreat') === 'move back' && resolveIntent(hurt, 'retreat') === 'turn right' && keyPress(hurt, 'retreat', 'turn right')[1].deg === 90, 'retreat backs off, turns when stuck');
  console.assert(resolveIntent({ ...hurt, x: 900 }, 'retreat') === 'move back', 'retreat backs off again after moving');
  resetNav();
  console.log('realdoom-logic.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
