// Shared plumbing for the three game-card renderers (render-snake.js, render-drive.js,
// render-doom.js / render-realdoom.js): design tokens (feeding the renderers' own canvas/
// THREE.js scenes - not reskinned here, out of scope), DOM chrome (HUD/buttons/decision
// strip), model-vs-scripted policy helpers, viewport pausing and keyboard override. Keeps the
// three renderers from re-deriving the same mount() contract three times.
//
// Chrome lives inside a `.media.dark > .media-scene` box (site/index.html): the canvas fills
// the scene, the HUD sits top-right (the `.media-label` owns top-left), the decision strip -
// a bars() readout (js/bars.js), same primitive as everywhere else on the page - docks
// bottom-left over a scrim, and the take-over hint + MODEL/RESTART controls dock bottom-right
// as small `.btn.outline`s. Chrome styles (.gc-*) live in exhibits.css.
import { bars } from '../bars.js';

// v11 dark-scene palette (designs/AGILITY_DESIGN.md lane). Keys kept so the renderers need no
// edits: putty = ground/field, ink = primary marks, bone = secondary objects, vellum =
// hairlines/grid, graphite = muted surfaces (road), paper = brightest accent.
export const TOKENS = {
  putty: '#0b0b0b',
  ink: '#f0eeeb',
  bone: '#938f89',
  vellum: '#2a2a2a',
  graphite: '#3a3a3a',
  paper: '#ffffff',
};

// Builds the shared DOM inside `el`: canvas stage (fills the box) + HUD (top-right) + decision
// strip/sentence (bottom-left, over a scrim) + hint/policy/restart controls (bottom-right).
// Returns refs the renderer draws into every frame/tick.
export function mountChrome(el, { label } = {}) {
  el.innerHTML = '';
  el.classList.add('gc-root');
  el.tabIndex = 0;

  const stage = document.createElement('div');
  stage.className = 'gc-stage';
  const canvas = document.createElement('canvas');
  stage.appendChild(canvas);

  const hud = document.createElement('div');
  hud.className = 'gc-hud t-eyebrow';
  hud.textContent = label ?? '';

  // decision strip + sentence, docked bottom-left over a scrim - refs.sentence.parentElement
  // is `foot`, which a renderer may append its own extra status lines into (same .gc-sentence
  // class), stacking below the read sentence.
  const foot = document.createElement('div');
  foot.className = 'gc-foot';
  const decision = document.createElement('div');
  decision.className = 'gc-decision';
  const sentence = document.createElement('div');
  sentence.className = 'gc-sentence';
  foot.append(decision, sentence);

  // hint + MODEL/RESTART, docked bottom-right - kept separate from `foot` so the two never
  // collide as the scrim's content grows.
  const footTop = document.createElement('div');
  footTop.className = 'gc-foot-top';
  const hint = document.createElement('div');
  hint.className = 'gc-hint';
  hint.textContent = '← → ↑ ↓ to take over · the model resumes after 3 s';
  const controls = document.createElement('div');
  controls.className = 'gc-controls';
  const policyBtn = document.createElement('button');
  policyBtn.className = 'btn outline gc-btn';
  policyBtn.type = 'button';
  const restartBtn = document.createElement('button');
  restartBtn.className = 'btn outline gc-btn';
  restartBtn.type = 'button';
  restartBtn.textContent = 'Restart';
  controls.append(policyBtn, restartBtn);
  footTop.append(hint, controls);

  el.append(stage, hud, foot, footTop);

  return { root: el, stage, canvas, hud, policyBtn, restartBtn, decision, sentence };
}

// "you vs model" HUD line — only shown once a human has taken at least one turn (a 0-0 line
// before any keypress would just be noise). `counts` is {you, model}; `unit` is an optional
// plural noun appended after the numbers (e.g. "kills"), omitted for a bare count.
export function scoreboardLine(counts, unit = '') {
  if (!counts || (counts.you === 0 && counts.model === 0)) return null;
  return `YOU ${counts.you} · MODEL ${counts.model}${unit ? ' ' + unit : ''}`;
}

// Paints the decision strip as a bars() readout (js/bars.js): candidate · ink-on-dark bar ·
// tabular value, plus the ∅ row, and the read sentence underneath. `probs` is a plain
// {label: p} map. Capped at 5 candidate rows (+ ∅) so Drive's longer rule lists never overflow
// the scene box. One bars() instance per game mount, reused across ticks via .update().
export function paintDecision(refs, { candidates = [], probs = {}, p_null = null, sentence = '' } = {}) {
  const rows = candidates.slice(0, 5).map((c) => ({ label: c, p: probs[c] ?? 0 }));
  if (p_null != null) rows.push({ label: '∅', p: p_null, isNull: true });
  if (!refs.decisionBars) refs.decisionBars = bars(refs.decision, rows);
  else refs.decisionBars.update(rows);
  refs.sentence.textContent = sentence;
}

// Fires onChange(visible) whenever `el`'s intersection with the viewport crosses 10%. Returns
// a disconnect function.
export function watchVisibility(el, onChange) {
  if (typeof IntersectionObserver === 'undefined') {
    onChange(true);
    return () => {};
  }
  const io = new IntersectionObserver(([entry]) => onChange(entry.isIntersecting), { threshold: 0.1 });
  io.observe(el);
  return () => io.disconnect();
}

// setInterval wrapper that can be paused (viewport) independent of being stopped (unmount).
export function createTicker(ms, fn) {
  let id = null;
  let paused = false;
  return {
    start() {
      if (id) return;
      id = setInterval(() => {
        if (!paused) fn();
      }, ms);
    },
    stop() {
      clearInterval(id);
      id = null;
    },
    pause() {
      paused = true;
    },
    resume() {
      paused = false;
    },
  };
}

// Tracks the most recent keyboard-driven move; policy loops check `.active()` each tick and
// play `.move` instead of calling the policy while a human is at the controls.
export function createHumanOverride(idleMs = 3000) {
  let move = null;
  let until = 0;
  return {
    set(m) {
      move = m;
      until = Date.now() + idleMs;
    },
    active() {
      return Date.now() < until;
    },
    get move() {
      return move;
    },
  };
}

// Binds keydown on `el` (only while focused) through a {key: action} map (keys lower-cased) to
// onAction(action). Returns an unbind function.
export function bindKeys(el, keymap, onAction) {
  const handler = (e) => {
    const action = keymap[e.key] ?? keymap[e.key.toLowerCase()];
    if (!action) return;
    e.preventDefault();
    onAction(action);
  };
  el.addEventListener('keydown', handler);
  return () => el.removeEventListener('keydown', handler);
}

// Single-request model policy: describe() the engine, offer its legal moves, argmax over
// non-null probability mass. Works for any engine exposing candidates()/safeMoves() +
// describe() (Snake, Doom, Drive all do). A 1-candidate choice is forced - skip the network
// round trip (server.py requires >=2 labels anyway).
export async function modelPolicy({ decide, engine, question }) {
  const labels = engine.candidates ? engine.candidates() : engine.safeMoves();
  if (labels.length === 0) return null;
  if (labels.length === 1) return { move: labels[0], probs: { [labels[0]]: 1 }, p_null: 0, ms: 0 };
  const t0 = performance.now();
  const res = await decide(engine.describe(), [{ type: 'choice', question, labels }]);
  const ms = performance.now() - t0;
  const r = res?.results?.[0];
  if (!r) return { move: labels[0], probs: {}, p_null: null, ms };
  const move = labels.reduce((best, m) => ((r.probs[m] ?? 0) > (r.probs[best] ?? 0) ? m : best), labels[0]);
  return { move, probs: r.probs, p_null: r.p_null, ms, device: res.device };
}

// Reads one recorded frame by index, looping back to 0 past the end - static-mode "MODEL"
// playback. Frame shape (scripts/record_games.mjs): {tick, state, desc, candidates, probs
// (array aligned to candidates), p_null, move, ms}.
export function replayFrame(frames, index) {
  if (!Array.isArray(frames) || frames.length === 0) return null;
  const i = index % frames.length;
  const f = frames[i];
  const probs = {};
  (f.candidates ?? []).forEach((label, idx) => {
    probs[label] = Array.isArray(f.probs) ? f.probs[idx] : (f.probs ?? {})[label];
  });
  return { ...f, probs, nextIndex: i + 1, looped: i + 1 >= frames.length };
}

function loadJSONCache() {
  const cache = new Map();
  return (url) => {
    if (!cache.has(url)) {
      cache.set(
        url,
        fetch(url)
          .then((res) => (res.ok ? res.json() : null))
          .catch(() => null)
      );
    }
    return cache.get(url);
  };
}
export const loadJSON = loadJSONCache();

function selfTest() {
  const ho = createHumanOverride(10);
  console.assert(!ho.active(), 'human override starts inactive');
  ho.set('up');
  console.assert(ho.active() && ho.move === 'up', 'set() activates override with the given move');

  const frames = [
    { tick: 0, candidates: ['a', 'b'], probs: [0.3, 0.7], p_null: 0.1, move: 'b' },
    { tick: 1, candidates: ['a', 'b'], probs: [0.9, 0.1], p_null: 0.0, move: 'a' },
  ];
  const f0 = replayFrame(frames, 0);
  console.assert(f0.move === 'b' && f0.probs.b === 0.7, 'replayFrame maps the probs array onto candidates');
  console.assert(scoreboardLine({ you: 0, model: 0 }) === null, 'scoreboardLine hides itself at 0-0');
  console.assert(scoreboardLine({ you: 3, model: 2 }) === 'YOU 3 · MODEL 2', 'scoreboardLine formats a bare count');
  console.assert(scoreboardLine({ you: 0, model: 1 }, 'kills') === 'YOU 0 · MODEL 1 kills', 'scoreboardLine appends the unit');

  const f2 = replayFrame(frames, 2);
  console.assert(f2.tick === 0, 'replayFrame loops back to the start past the end');

  console.log('loop.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
