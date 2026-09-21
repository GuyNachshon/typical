// Shared plumbing for the three game-card renderers (render-snake.js, render-drive.js,
// render-doom.js): monochrome design tokens, DOM chrome (HUD/buttons/decision strip),
// model-vs-scripted policy helpers, viewport pausing and keyboard override. Keeps the three
// renderers from re-deriving the same mount() contract three times.

export const TOKENS = {
  putty: '#c4c3b6',
  ink: '#000000',
  bone: '#e7e5e4',
  vellum: '#dfdcd5',
  graphite: '#595855',
  paper: '#ffffff',
};

let stylesInjected = false;
// Idempotent - safe to call from every renderer's mount(), only the first call does anything.
function injectChromeStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const style = document.createElement('style');
  style.id = 'gc-chrome-styles';
  style.textContent = `
.gc-root { position:relative; width:100%; height:100%; display:flex; flex-direction:column; background:${TOKENS.ink}; color:${TOKENS.paper}; font-family:'Inter',sans-serif; outline:none; }
.gc-stage { position:relative; flex:1 1 auto; min-height:0; overflow:hidden; }
.gc-stage canvas { position:absolute; inset:0; width:100%; height:100%; display:block; }
.gc-hud { position:absolute; top:10px; left:12px; font-size:9px; letter-spacing:.08em; text-transform:uppercase; color:${TOKENS.paper}; opacity:.85; pointer-events:none; z-index:3; }
.gc-hud div { margin-top:2px; }
.gc-controls { position:absolute; bottom:10px; right:10px; display:flex; gap:6px; z-index:3; }
.gc-btn { font-family:inherit; font-size:8px; letter-spacing:.08em; text-transform:uppercase; color:${TOKENS.paper}; background:transparent; border:1px solid ${TOKENS.graphite}; border-radius:2px; padding:4px 8px; cursor:pointer; opacity:.8; }
.gc-btn:hover { opacity:1; border-color:${TOKENS.bone}; }
.gc-btn.gc-on { background:${TOKENS.paper}; color:${TOKENS.ink}; opacity:1; }
.gc-foot { flex:0 0 auto; border-top:1px solid ${TOKENS.graphite}; padding:8px 12px; }
.gc-decision { display:flex; flex-direction:column; gap:2px; margin-bottom:6px; }
.gc-drow { display:grid; grid-template-columns:74px 1fr 30px; align-items:center; gap:6px; font-size:8px; letter-spacing:.03em; text-transform:uppercase; color:${TOKENS.bone}; }
.gc-dtrack { height:6px; background:${TOKENS.graphite}; position:relative; }
.gc-dbar { position:absolute; left:0; top:0; bottom:0; background:${TOKENS.paper}; }
.gc-dnum { font-family:'Instrument Serif',Georgia,serif; font-size:11px; color:${TOKENS.paper}; text-align:right; }
.gc-sentence { font-size:9px; line-height:1.4; color:${TOKENS.graphite}; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.gc-root:focus .gc-stage { box-shadow: inset 0 0 0 1px ${TOKENS.bone}; }
`;
  document.head.appendChild(style);
}

// Builds the shared DOM inside `el`: canvas + HUD + policy/restart buttons + decision strip +
// read sentence. Returns refs the renderer draws into every frame/tick.
export function mountChrome(el, { label } = {}) {
  injectChromeStyles();
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
  policyBtn.className = 'gc-btn';
  policyBtn.type = 'button';
  const restartBtn = document.createElement('button');
  restartBtn.className = 'gc-btn';
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

// Paints the candidate · ink bar · serif numeral list + the ∅ row, and the read sentence
// underneath. `probs` is a plain {label: p} map.
export function paintDecision(refs, { candidates = [], probs = {}, p_null = null, sentence = '' } = {}) {
  refs.decision.innerHTML = '';
  const rows = candidates.map((c) => [c, probs[c] ?? 0]);
  if (p_null != null) rows.push(['∅ (null)', p_null]);
  for (const [label, p] of rows) {
    const row = document.createElement('div');
    row.className = 'gc-drow';
    const name = document.createElement('span');
    name.textContent = label;
    const track = document.createElement('span');
    track.className = 'gc-dtrack';
    const bar = document.createElement('span');
    bar.className = 'gc-dbar';
    bar.style.width = `${Math.max(0, Math.min(1, p)) * 100}%`;
    track.appendChild(bar);
    const num = document.createElement('span');
    num.className = 'gc-dnum';
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
