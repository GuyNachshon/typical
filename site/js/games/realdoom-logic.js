// Pure grammar/policy helpers for real DOOM (js/games/render-realdoom.js and
// scripts/record_realdoom.mjs both import this - no DOM, safe in Node and the browser). Kept
// separate from render-realdoom.js so the recorder can drive the exact same sentence grammar
// and legal-move rules the live card uses, without dragging THREE.js/loop.js into Node.
const CELL_UNITS = 64; // map units per Doom grid cell (doom-wasm-notes.md)
const CROSSHAIR_DEG = 8;
const BEHIND_DEG = 100;

export const ACTIONS = ['move forward', 'move back', 'turn left', 'turn right', 'shoot'];

// Chocolate Doom key id + hold duration (ms) for each candidate label - press(key, ms) on
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
  if (Math.abs(b) > BEHIND_DEG) return 'behind';
  if (m.in_crosshair || Math.abs(b) <= CROSSHAIR_DEG) return 'ahead in the crosshair';
  return b < -CROSSHAIR_DEG ? 'to the left' : 'to the right';
}

// state -> the legal action set at 0 ammo (no shoot) and blocked_ahead (no move forward),
// canonical ACTIONS order.
export function candidatesFor(state) {
  return ACTIONS.filter((a) => {
    if (a === 'shoot' && (state.ammo ?? 0) <= 0) return false;
    if (a === 'move forward' && state.blocked_ahead) return false;
    return true;
  });
}

// state -> the sentence the model reads (demo-spec-v2.md #5's grammar, extended with the
// real state's monster type/dist/bearing fields instead of the ASCII-arena's cell counts).
export function describeDoom(state) {
  const sentences = [`Health ${state.health}, ammo ${state.ammo}.`];
  const monsters = (state.monsters ?? []).slice().sort((a, b) => a.dist - b.dist);
  monsters.forEach((m, i) => {
    const label = i === 0 ? 'An enemy' : 'Another enemy';
    const d = cellsOf(m.dist);
    sentences.push(`${label} (${m.type}) is ${d} cell${d === 1 ? '' : 's'} ${bearingPhrase(m)}.`);
  });
  if (state.blocked_ahead) sentences.push('A wall is ahead.');
  sentences.push(`Legal actions: ${candidatesFor(state).join(', ')}.`);
  return sentences.join(' ');
}

// Scripted policy (never calls the model): fire on a crosshair target within 20 cells and
// ammo > 0; else turn toward the nearest monster; else forward unless blocked, then turn right.
export function scriptedPolicy(state, legal) {
  const monsters = state.monsters ?? [];
  const target = monsters.find((m) => cellsOf(m.dist) <= 20 && (m.in_crosshair || Math.abs(m.bearing ?? 0) <= CROSSHAIR_DEG));
  if (target && legal.includes('shoot')) return 'shoot';
  if (monsters.length) {
    const nearest = [...monsters].sort((a, b) => a.dist - b.dist)[0];
    if ((nearest.bearing ?? 0) < 0 && legal.includes('turn left')) return 'turn left';
    if ((nearest.bearing ?? 0) > 0 && legal.includes('turn right')) return 'turn right';
  }
  if (legal.includes('move forward')) return 'move forward';
  if (legal.includes('turn right')) return 'turn right';
  return legal[0] ?? null;
}

function selfTest() {
  const s1 = { health: 84, ammo: 40, blocked_ahead: false, monsters: [{ type: 'zombieman', dist: 320, bearing: 3, in_crosshair: true }] };
  console.assert(describeDoom(s1) === 'Health 84, ammo 40. An enemy (zombieman) is 5 cells ahead in the crosshair. Legal actions: move forward, move back, turn left, turn right, shoot.', 'crosshair sentence + legal actions');
  console.assert(scriptedPolicy(s1, candidatesFor(s1)) === 'shoot', 'scripted fires on a crosshair target in range');

  const s2 = { health: 20, ammo: 0, blocked_ahead: true, monsters: [{ type: 'imp', dist: 700, bearing: -40, in_crosshair: false }] };
  console.assert(describeDoom(s2) === 'Health 20, ammo 0. An enemy (imp) is 11 cells to the left. A wall is ahead. Legal actions: move back, turn left, turn right.', 'left bearing + blocked + no-shoot');
  console.assert(!candidatesFor(s2).includes('shoot'), 'shoot dropped at 0 ammo');
  console.assert(!candidatesFor(s2).includes('move forward'), 'move forward dropped when blocked');
  console.assert(scriptedPolicy(s2, candidatesFor(s2)) === 'turn left', 'scripted turns toward a nearer-left monster out of shoot range/angle');

  const s3 = { health: 100, ammo: 50, blocked_ahead: false, monsters: [] };
  console.assert(describeDoom(s3) === 'Health 100, ammo 50. Legal actions: move forward, move back, turn left, turn right, shoot.', 'no monsters -> no enemy sentence');
  console.assert(scriptedPolicy(s3, candidatesFor(s3)) === 'move forward', 'scripted advances with nothing in view');

  console.log('realdoom-logic.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
