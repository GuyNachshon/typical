// 02 DOOM demo screen. Honest failure exhibit: the model says "shoot" in 29 of 30 probed
// situations (demo-spec-v2.md #5), enemy or no enemy. The live loop keeps running anyway so
// visitors see it happen; a greedy toggle shows what a scripted policy does instead.
import { Doom, greedyPolicy } from '../games/doom.js';
import { thermal } from '../thermal.js';

const SEED = 7;
const TICK_MS = 250;

// Representation B from demo-spec-v2.md #5 (the shipped/probed one, .200 acc, chance .200) -
// exact string, sent verbatim as the query.
export const QUESTION =
  "Which action applies for the player? Apply the rules in the stated precedence order; the first matching rule wins.\n" +
  'move back: applies when health is below 30 and an enemy is visible  shoot: applies when an enemy is in the crosshair and ammo is above 0  turn left: applies when an enemy is to the left, outside the crosshair  turn right: applies when an enemy is to the right, outside the crosshair  move forward: applies when none of the above rules fire';

// char -> css color for the raycast view + minimap glyphs, thermal ramp by depth (wall shade
// chars) or by object type (minimap). null = default foreground, no span needed.
function glyphColor(ch) {
  if (ch === '█') return thermal(0.95); // near wall / close enemy blob
  if (ch === '▓') return thermal(0.72);
  if (ch === '▒') return thermal(0.5);
  if (ch === '░') return thermal(0.28);
  if (ch === '·') return thermal(0.12); // far wall / far enemy dot / floor dither
  if (ch === '¥') return thermal(0.85); // mid-range enemy column
  if (ch === 'a') return thermal(0.55); // ammo pickup
  if (ch === '+') return thermal(0.92); // crosshair / health pickup
  if (ch === 'E') return thermal(0.9); // minimap enemy
  if (ch === '#' || ch === 'O') return 'var(--dim)'; // minimap wall / pillar
  if (ch === 'D') return thermal(0.4); // minimap door
  if (ch === '▲' || ch === '▶' || ch === '▼' || ch === '◀') return thermal(1); // player arrow
  if (ch === '.') return 'var(--dim)';
  return null;
}

// Run-length-encodes consecutive same-color chars into <span>s so a 16x98 frame at 4fps
// doesn't churn ~1500 DOM nodes a second. Glyph alphabet here has no HTML-special chars.
function paint(pre, text) {
  let html = '';
  let curColor;
  let buf = '';
  const flush = () => {
    if (!buf) return;
    html += curColor ? `<span style="color:${curColor}">${buf}</span>` : buf;
    buf = '';
  };
  for (const ch of text) {
    const c = ch === '\n' ? null : glyphColor(ch);
    if (c !== curColor) {
      flush();
      curColor = c;
    }
    buf += ch;
  }
  flush();
  pre.innerHTML = html;
}

function button(label) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'btn';
  b.textContent = label;
  return b;
}

let replayPromise = null;
function loadReplay() {
  if (!replayPromise) {
    replayPromise = fetch('data/replays/doom.json')
      .then((res) => (res.ok ? res.json() : null))
      .catch(() => null);
  }
  return replayPromise;
}

export async function mount(el, ctx) {
  el.innerHTML = '';
  el.classList.add('snake-screen');

  const left = document.createElement('div');
  const pre = document.createElement('pre');
  pre.style.cssText = 'font-family:var(--mono);white-space:pre;font-size:1.05rem;line-height:1.15;margin:0;color:var(--fg);letter-spacing:0';
  const stateEl = document.createElement('div');
  stateEl.className = 'snake-state';
  left.append(pre, stateEl);

  const right = document.createElement('div');
  const fieldHost = document.createElement('div');
  fieldHost.className = 'snake-bars-field';
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'width:100%;height:100%';
  fieldHost.appendChild(canvas);
  const hud = document.createElement('div');
  hud.className = 'snake-hud';
  const tally = document.createElement('div');
  tally.className = 'dim caption';
  tally.style.marginTop = '0.35rem';
  const controls = document.createElement('div');
  controls.style.cssText = 'display:flex;gap:0.5rem;margin-top:0.75rem';
  const btnModel = button('model');
  const btnGreedy = button('greedy');
  const btnRestart = button('restart');
  btnModel.setAttribute('aria-pressed', 'true');
  btnGreedy.setAttribute('aria-pressed', 'false');
  controls.append(btnModel, btnGreedy, btnRestart);
  right.append(fieldHost, hud, tally, controls);

  el.append(left, right);

  const field = ctx.heatField(canvas, { rows: [] });
  const sectionEl = el.closest('.screen') ?? el;

  let engine = new Doom({ seed: SEED });
  let mode = 'model'; // 'model' | 'greedy'
  let replayTick = 0;
  let shots = 0;
  let shotsCrosshair = 0;
  let hits = 0;
  let running = false;
  let timer = null;

  function resetTally() {
    shots = 0;
    shotsCrosshair = 0;
    hits = 0;
  }

  function restart() {
    engine = new Doom({ seed: SEED });
    replayTick = 0;
    resetTally();
    draw({});
  }

  async function decideMove() {
    const cands = engine.candidates();
    if (cands.length === 0) return null;
    if (ctx.mode() === 'static') {
      const frames = await loadReplay();
      if (!Array.isArray(frames) || frames.length === 0) return { move: greedyPolicy(engine), probs: {}, ms: 0 };
      if (replayTick >= frames.length) replayTick = 0;
      const f = frames[replayTick++];
      const probs = {};
      (f.candidates ?? cands).forEach((label, i) => {
        probs[label] = Array.isArray(f.probs) ? f.probs[i] : (f.probs ?? {})[label];
      });
      return { move: f.move, probs, p_null: f.p_null, ms: f.ms, device: 'mps (recorded)' };
    }
    const t0 = performance.now();
    const res = await ctx.decide(engine.describe(), [{ type: 'choice', question: QUESTION, labels: cands }]);
    const ms = performance.now() - t0;
    const r = res?.results?.[0];
    if (!r) return { move: greedyPolicy(engine), probs: {}, ms };
    const move = cands.reduce((best, m) => ((r.probs[m] ?? 0) > (r.probs[best] ?? 0) ? m : best), cands[0]);
    return { move, probs: r.probs, p_null: r.p_null, ms, device: res.device };
  }

  function draw(result) {
    paint(pre, engine.render());
    stateEl.textContent = engine.describe();
    const cands = engine.candidates();
    const rows = cands.map((label) => ({ label, p: result.probs?.[label] ?? 0 }));
    field.update(rows, result.p_null ?? (mode === 'greedy' ? 0 : null));
    const p = engine.player;
    hud.textContent = `health ${p.health} · ammo ${p.ammo} · score ${engine.score} · ticks ${engine.ticks} · ${mode}`;
    tally.textContent = `shots fired ${shots} · enemy in crosshair ${shotsCrosshair} · hits ${hits}`;
    if (result.ms != null || result.device) ctx.readout(sectionEl, { ms: result.ms, device: result.device });
  }

  async function tick() {
    if (!running) return;
    if (engine.dead || engine.candidates().length === 0) {
      restart();
      schedule();
      return;
    }
    let result;
    try {
      result = mode === 'greedy' ? { move: greedyPolicy(engine), probs: {}, ms: 0 } : await decideMove();
    } catch (err) {
      console.error('doom tick error', err);
      schedule();
      return;
    }
    if (!running) return; // screen may have left the viewport while awaiting decideMove
    if (!result || !result.move) {
      restart();
      schedule();
      return;
    }
    const crosshair = engine.describe().includes('in the crosshair');
    const scoreBefore = engine.score;
    engine.step(result.move);
    if (result.move === 'shoot') {
      shots += 1;
      if (crosshair) shotsCrosshair += 1;
      if (engine.score > scoreBefore) hits += 1;
    }
    draw(result);
    schedule();
  }

  function schedule() {
    if (!running) return;
    timer = setTimeout(tick, TICK_MS);
  }

  function start() {
    if (running) return;
    running = true;
    tick();
  }

  function stop() {
    running = false;
    clearTimeout(timer);
  }

  btnModel.addEventListener('click', () => {
    mode = 'model';
    btnModel.setAttribute('aria-pressed', 'true');
    btnGreedy.setAttribute('aria-pressed', 'false');
  });
  btnGreedy.addEventListener('click', () => {
    mode = 'greedy';
    btnModel.setAttribute('aria-pressed', 'false');
    btnGreedy.setAttribute('aria-pressed', 'true');
  });
  btnRestart.addEventListener('click', restart);

  draw({});

  // Only one loop runs at a time per screen; stop entirely once this screen scrolls out so
  // it never hammers the shared single-GPU-slot server alongside another live game.
  new IntersectionObserver((entries) => {
    const vis = entries.some((e) => e.isIntersecting);
    console.log('doom io', vis, entries.map((e) => e.intersectionRatio));
    vis ? start() : stop();
  }).observe(sectionEl);
}
