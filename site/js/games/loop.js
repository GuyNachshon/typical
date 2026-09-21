// Shared plumbing for the three game-card renderers (render-snake.js, render-drive.js,
// render-doom.js): design tokens (feeding the renderers' own canvas/THREE.js scenes - not
// reskinned here, out of scope), DOM chrome (HUD/buttons/decision strip), model-vs-scripted
// policy helpers, viewport pausing and keyboard override. Keeps the three renderers from
// re-deriving the same mount() contract three times. Chrome styles (.gc-*) live in
// exhibits.css, which reuses the .ex-row/.ex-bar/.ex-ghost kit for the decision strip and
// policy/restart controls - see the "game chrome" section there.
// Atoms palette (keys kept so the renderers need no edits): canvas black, cream strokes,
// champagne for the one emphasised element, ash for muted surfaces.
export const TOKENS = {
  putty: '#000000',   // field / ground
  ink: '#fff7dd',     // primary marks (snake body, ego car, wall edges)
  bone: '#8f8b83',    // secondary objects (traffic, floor)
  vellum: '#2a2825',  // hairlines / grid
  graphite: '#66635f',// muted surfaces (road)
  paper: '#fff7dd',
};

// Builds the shared DOM inside `el`: canvas + HUD + policy/restart buttons + decision strip +
// read sentence. Returns refs the renderer draws into every frame/tick.
export function mountChrome(el, { label } = {}) {
  el.innerHTML = '';
  el.classList.add('gc-root');
  el.tabIndex = 0;

  const stage = document.createElement('div');
  stage.className = 'gc-stage';
  const canvas = document.createElement('canvas');
  stage.appendChild(canvas);

  const hud = document.createElement('div');
  hud.className = 'gc-hud';
  hud.textContent = label ?? '';

  const controls = document.createElement('div');
  controls.className = 'gc-controls';
  const policyBtn = document.createElement('button');
  policyBtn.className = 'ex-ghost gc-btn';
  policyBtn.type = 'button';
  const restartBtn = document.createElement('button');
  restartBtn.className = 'ex-ghost gc-btn';
  restartBtn.type = 'button';
  restartBtn.textContent = 'Restart';
  controls.append(policyBtn, restartBtn);

  stage.append(hud, controls);

  const foot = document.createElement('div');
  foot.className = 'gc-foot';
  const decision = document.createElement('div');
  decision.className = 'gc-decision';
  const sentence = document.createElement('div');
  sentence.className = 'gc-sentence';
  foot.append(decision, sentence);

  el.append(stage, foot);

  return { root: el, stage, canvas, hud, policyBtn, restartBtn, decision, sentence };
}

// Paints the candidate · cream bar · tabular numeral list + the ∅ row, and the read sentence
// underneath, reusing the .ex-row/.ex-bar kit (exhibits.css) so the decision strip matches
// every data exhibit. `probs` is a plain {label: p} map. The highest-p row (candidate or ∅)
// gets the champagne fill - it's the winner.
export function paintDecision(refs, { candidates = [], probs = {}, p_null = null, sentence = '' } = {}) {
  refs.decision.innerHTML = '';
  const rows = candidates.map((c) => [c, probs[c] ?? 0]);
  if (p_null != null) rows.push(['∅ (null)', p_null]);
  const maxP = rows.reduce((m, [, p]) => Math.max(m, p), -Infinity);
  for (const [label, p] of rows) {
    const row = document.createElement('div');
    row.className = 'ex-row';
    const name = document.createElement('span');
    name.className = 'ex-row-text';
    name.textContent = label;
    const track = document.createElement('span');
    track.className = 'ex-bar';
    const bar = document.createElement('span');
    bar.className = 'ex-bar-fill' + (p === maxP ? ' is-winner' : '');
    bar.style.width = `${Math.max(0, Math.min(1, p)) * 100}%`;
    track.appendChild(bar);
    const num = document.createElement('span');
    num.className = 'ex-num';
    num.textContent = p.toFixed(2);
    row.append(name, track, num);
    refs.decision.appendChild(row);
  }
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
  const f2 = replayFrame(frames, 2);
  console.assert(f2.tick === 0, 'replayFrame loops back to the start past the end');

  console.log('loop.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
