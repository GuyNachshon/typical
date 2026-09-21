// Pure grammar/policy helpers for real DOOM (js/games/render-realdoom.js and
// scripts/record_realdoom.mjs both import this - no DOM, safe in Node and the browser). Kept
// separate from render-realdoom.js so the recorder can drive the exact same sentence grammar
// and legal-move rules the live card uses, without dragging THREE.js/loop.js into Node.
// Same demo-spec v3 design as js/games/doom.js: the model picks an intent (retreat / engage /
// explore) from the situations that apply; resolveIntent() aims and picks the key.
import { INTENTS, HURT_BELOW } from './doom.js';

const CELL_UNITS = 64; // map units per Doom grid cell (doom-wasm-notes.md)
const CROSSHAIR_DEG = 8;
const BEHIND_DEG = 100; // |bearing| beyond this = behind the player = not "in sight"
const SIGHT_CELLS = 40; // Doom.state() lists the nearest monsters map-wide; "in sight" = line of sight (visible), not behind, this close

export const ACTIONS = ['move forward', 'move back', 'turn left', 'turn right', 'shoot'];

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
  'turn left': ['left', 180],
  'turn right': ['right', 180],
  shoot: ['fire', 120],
};

export function cellsOf(units) {
  return Math.max(0, Math.round(units / CELL_UNITS));
}

function bearingPhrase(m) {
  const b = m.bearing ?? 0;
  if (m.in_crosshair || Math.abs(b) <= CROSSHAIR_DEG) return 'ahead in the crosshair';
  return b < 0 ? 'to the left' : 'to the right';
}

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

// The model's candidates: the three intents, always.
export function candidatesFor(state) {
  return state.in_level === false ? [] : [...INTENTS];
}

// state -> the sentence the model reads: health/ammo, then only the situations that apply,
// worded exactly as the rule conditions (js/games/doom.js RULES), monster type included.
export function describeDoom(state) {
  const sentences = [`The player has ${state.health} health and ${state.ammo} ammo.`];
  if (state.health < HURT_BELOW) sentences.push('The player is badly hurt.');
  const seen = inSight(state);
  const where = (m) => {
    const d = cellsOf(m.dist);
    return `${/^[aeiou]/i.test(m.type) ? 'an' : 'a'} ${m.type} ${d} cell${d === 1 ? '' : 's'} ${bearingPhrase(m)}`;
  };
  if (seen.length === 1) sentences.push(`The player has an enemy in sight: ${where(seen[0])}.`);
  else if (seen.length > 1) sentences.push(`The player has ${seen.length} enemies in sight; the nearest is ${where(seen[0])}.`);
  else sentences.push('The player sees no enemy.');
  if (state.blocked_ahead) sentences.push('A wall is ahead.');
  return sentences.join(' ');
}

// explore navigator: forward + wall-follow, plus a stuck-timer escape. Pure wall-following
// (fixed turn preference) loops around whichever room it starts in almost indefinitely on
// E1M1 - state.monsters carries dist/bearing for the nearest thinkers map-wide even when
// they're behind walls (P_CheckSight/visible is what's withheld from the model's sentence,
// not from the engine), so a wall-hit turns toward the nearest one's side instead of a fixed
// direction. That's still "the engine aims", not the model - it only decides the intent.
// State is module-level (one page, one player) - resetNav() clears it on restart.
let navBias = 1; // 1 = right, -1 = left; the stuck-escape direction, flips when stuck
let navHist = []; // {x, y, t} samples since navWindowSince - a corner where the player
// alternates turnLock direction can still cover >32 units step to step (bouncing between two
// points a wall-width apart), so "progress" is a bounding box over a fixed time window, not
// distance from the last sample.
let navWindowSince = null; // when the current STUCK_MS window started
let navForce = 0; // ticks left in a forced stuck-escape turn
let navForceFwd = 0; // ticks left in the forced-forward burst that follows the turn
let turnLock = 0; // 0 outside a wall encounter, else the direction (1/-1) picked for its whole duration
const STUCK_MS = 2500;
const STUCK_RADIUS = 150; // bounding-box diagonal of the last STUCK_MS of positions, below this = no real progress
// A tight pillar room (E1M1 has one) can bounce the plain wall-follow turn back and forth
// between two spots a wall-width apart forever - the escape needs to be decisive: ~half a
// turn (not a quarter) plus a forced walk out of the pocket, not just a re-aim in place.
const FORCE_TURN_TICKS = 8; // ~8 * KEY_FOR_MOVE['turn right'][1] (180ms) holds ~= a 144 degree turn
const FORCE_FWD_TICKS = 4; // ~4 * KEY_FOR_MOVE['move forward'][1] (350ms) holds - walk out of the pocket

export function resetNav() {
  navBias = 1;
  navHist = [];
  navWindowSince = null;
  navForce = 0;
  navForceFwd = 0;
  turnLock = 0;
}

function exploreAction(state, now = Date.now()) {
  if (navWindowSince == null) navWindowSince = now;
  navHist.push({ x: state.x, y: state.y, t: now });
  // Check once per fixed STUCK_MS window, not "trim to the last STUCK_MS then compare" - trimming
  // first always leaves the oldest sample younger than STUCK_MS, so that check never fires.
  if (navForce === 0 && navForceFwd === 0 && now - navWindowSince >= STUCK_MS) {
    const xs = navHist.map((p) => p.x);
    const ys = navHist.map((p) => p.y);
    const span = Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    navWindowSince = now;
    navHist = [{ x: state.x, y: state.y, t: now }];
    if (span < STUCK_RADIUS) {
      // A fixed alternation (always flip) can settle into an exact repeating loop around a
      // tight obstacle (E1M1's pillar room does this) - a random side plus a randomised turn
      // length breaks that periodicity instead of retracing the same failed escape every time.
      navBias = Math.random() < 0.5 ? 1 : -1;
      navForce = FORCE_TURN_TICKS + Math.floor(Math.random() * 5); // 8-12 ticks, ~144-216 degrees
      turnLock = 0; // the stuck escape overrides whatever the wall encounter had picked
    }
  }
  if (navForce > 0) {
    navForce -= 1;
    if (navForce === 0) navForceFwd = FORCE_FWD_TICKS;
    return navBias > 0 ? 'turn right' : 'turn left';
  }
  if (navForceFwd > 0) {
    navForceFwd -= 1;
    return 'move forward'; // committed - blocked_ahead resumes governing once this burst ends
  }
  if (state.blocked_ahead) {
    // Pick (and keep) one turn direction for this whole wall encounter - recomputing it every
    // tick from the target's bearing oscillates left/right forever right at a corner, where the
    // bearing sign flips as the player's own angle changes.
    if (turnLock === 0) {
      const nearest = (state.monsters ?? []).reduce((best, m) => (best == null || m.dist < best.dist ? m : best), null);
      turnLock = nearest ? (((nearest.bearing ?? 0) >= 0) ? 1 : -1) : navBias;
    }
    return turnLock > 0 ? 'turn right' : 'turn left';
  }
  turnLock = 0;
  return 'move forward';
}

// Intent -> key press (the engine aims; the page says so). Raw actions pass through for the
// human override.
export function resolveIntent(state, move) {
  if (!INTENTS.includes(move)) return move;
  if (move === 'retreat') return 'move back';
  const target = inSight(state)[0];
  if (move === 'engage' && target) {
    const b = target.bearing ?? 0;
    if (target.in_crosshair || Math.abs(b) <= CROSSHAIR_DEG) return (state.ammo ?? 0) > 0 ? 'shoot' : 'move forward';
    return b < 0 ? 'turn left' : 'turn right';
  }
  if (move === 'explore') return exploreAction(state);
  return state.blocked_ahead ? 'turn right' : 'move forward';
}

// Scripted policy = RULES applied literally (never calls the model) -> an intent.
export function scriptedPolicy(state) {
  if (state.health < HURT_BELOW) return 'retreat';
  return inSight(state).length ? 'engage' : 'explore';
}

function selfTest() {
  const s1 = { health: 84, ammo: 40, blocked_ahead: false, monsters: [{ type: 'zombieman', dist: 320, bearing: 3, in_crosshair: true }] };
  console.assert(describeDoom(s1) === 'The player has 84 health and 40 ammo. The player has an enemy in sight: a zombieman 5 cells ahead in the crosshair.', 'crosshair sentence');
  console.assert(candidatesFor(s1).join() === 'retreat,engage,explore', 'candidates are the intents');
  console.assert(scriptedPolicy(s1) === 'engage' && resolveIntent(s1, 'engage') === 'shoot', 'engage on a crosshair target -> shoot');

  const s2 = { health: 20, ammo: 0, blocked_ahead: true, monsters: [{ type: 'imp', dist: 700, bearing: -40, in_crosshair: false }, { type: 'demon', dist: 3000, bearing: 10, in_crosshair: false }, { type: 'zombieman', dist: 500, bearing: 2, in_crosshair: true, visible: false }] };
  console.assert(describeDoom(s2) === 'The player has 20 health and 0 ammo. The player is badly hurt. The player has an enemy in sight: an imp 11 cells to the left. A wall is ahead.', 'hurt + left bearing + far monster and wall-hidden monster dropped + wall');
  console.assert(scriptedPolicy(s2) === 'retreat' && resolveIntent(s2, 'retreat') === 'move back', 'badly hurt -> retreat -> move back');
  console.assert(resolveIntent(s2, 'engage') === 'turn left', 'engage on a left monster -> turn left');
  console.assert(resolveIntent(s2, 'explore') === 'turn right', 'explore into a wall -> turn right');
  console.assert(resolveIntent(s2, 'turn right') === 'turn right', 'raw actions pass through');

  const s3 = { health: 100, ammo: 50, blocked_ahead: false, monsters: [{ type: 'imp', dist: 500, bearing: 160, in_crosshair: false }] };
  console.assert(describeDoom(s3) === 'The player has 100 health and 50 ammo. The player sees no enemy.', 'a monster behind is not in sight');
  console.assert(scriptedPolicy(s3) === 'explore' && resolveIntent(s3, 'explore') === 'move forward', 'scripted advances with nothing in sight');

  const s4 = { health: 60, ammo: 8, blocked_ahead: false, monsters: [{ type: 'imp', dist: 400, bearing: 30, in_crosshair: false }, { type: 'zombieman', dist: 600, bearing: -2, in_crosshair: true }] };
  console.assert(describeDoom(s4) === 'The player has 60 health and 8 ammo. The player has 2 enemies in sight; the nearest is a zombieman 9 cells ahead in the crosshair.', 'crosshair monster leads even if farther');

  console.log('realdoom-logic.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
