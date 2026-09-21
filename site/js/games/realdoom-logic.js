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
  'move forward': ['forward', 350],
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

// Intent -> key press (the engine aims; the page says so). Raw actions pass through for the
// human override.
export function resolveIntent(state, move) {
  if (!INTENTS.includes(move)) return move;
  const forward = state.blocked_ahead ? 'turn right' : 'move forward';
  if (move === 'retreat') return 'move back';
  const target = inSight(state)[0];
  if (move === 'engage' && target) {
    const b = target.bearing ?? 0;
    if (target.in_crosshair || Math.abs(b) <= CROSSHAIR_DEG) return (state.ammo ?? 0) > 0 ? 'shoot' : forward;
    return b < 0 ? 'turn left' : 'turn right';
  }
  return forward;
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
