// 03 DRIVE demo screen. demo-spec-v2.md #4: facts = state, Choice K=6 with a precedence rule
// list. Ships with a guardrail (engine.candidates() only offers manoeuvres that exist) and an
// honest running tally, because the probe found it never once chooses to change lanes.
import { Drive, greedyPolicy } from '../games/drive.js';
import { thermal } from '../thermal.js';

const SEED = 7;
const TICK_MS = 250;

// Variant B from demo-spec-v2.md #4 (the shipped one, .533 acc, chance .167) - exact string.
export const QUESTION =
  'Which manoeuvre applies? Apply the rules in the stated precedence order; the first matching rule wins.\n' +
  "stop: applies when a pedestrian is crossing the car's lane within 20 m  brake: applies when the traffic light ahead is red within 60 m, or when the lead vehicle is slower than the car and within 20 m and no adjacent lane is clear  change lane right: applies when the destination exit is on the right within 300 m and the right lane is clear, or when the lead vehicle is slower and within 20 m and the right lane is clear  change lane left: applies when the lead vehicle is slower and within 20 m and the left lane is clear and the right lane is not clear  accelerate: applies when the car is below the speed limit, no vehicle is within 50 m ahead, and no red light is ahead  hold speed: applies when none of the above rules fire";

const RED_LIGHT_RANGE = 60; // m - matches the brake rule's threshold in QUESTION
const PED_RANGE = 20; // m - matches the stop rule's threshold in QUESTION

function classifyLine(line) {
  if (line.includes('RED')) return 'red';
  if (line.includes('GREEN')) return 'green';
  if (line.includes('◟') || line.includes('◙')) return 'ego'; // ▟ ▙
  if (line.includes('◛') || line.includes('◜')) return 'traffic'; // ▛ ▜
  return 'road';
}

function glyphColor(ch, cls) {
  if ((cls === 'red' || cls === 'green') && ch.trim()) return cls === 'red' ? thermal(0.3) : 'var(--live)';
  if (cls === 'ego' && '◟█◙'.includes(ch)) return thermal(1); // ▟█▙
  if (cls === 'traffic' && '◛█◜'.includes(ch)) return thermal(0.6); // ▛█▜
  if (ch === '☺') return thermal(0.85); // ☺ pedestrian
  if (ch === '╱') return thermal(0.45); // ╱ exit ramp
  if ('║┆¦━'.includes(ch)) return 'var(--dim)'; // ║ ┆ ¦ ━
  return null;
}

// Run-length-encodes consecutive same-color chars per line into <span>s. Glyph alphabet here
// has no HTML-special chars.
function paint(pre, text) {
  const lines = text.split('\n');
  let html = '';
  lines.forEach((line, li) => {
    const cls = classifyLine(line);
    let curColor;
    let buf = '';
    const flush = () => {
      if (!buf) return;
      html += curColor ? `<span style="color:${curColor}">${buf}</span>` : buf;
      buf = '';
    };
    for (const ch of line) {
      const c = glyphColor(ch, cls);
      if (c !== curColor) {
        flush();
        curColor = c;
      }
      buf += ch;
    }
    flush();
    if (li < lines.length - 1) html += '\n';
  });
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
    replayPromise = fetch('data/replays/drive.json')
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
  pre.style.cssText = 'font-family:var(--mono);white-space:pre;font-size:1.05rem;line-height:1.05;margin:0;color:var(--fg);letter-spacing:0';
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

  let engine = new Drive({ seed: SEED });
  let mode = 'model'; // 'model' | 'greedy'
  let replayTick = 0;
  let redTotal = 0;
  let redRespected = 0;
  let pedTotal = 0;
  let pedStopped = 0;
  let laneChangeTotal = 0;
  let laneChangeChosen = 0;
  let running = false;
  let timer = null;

  function resetTally() {
    redTotal = 0;
    redRespected = 0;
    pedTotal = 0;
    pedStopped = 0;
    laneChangeTotal = 0;
    laneChangeChosen = 0;
  }

  function restart() {
    engine = new Drive({ seed: SEED });
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
    const e = engine.ego;
    hud.textContent = `${e.speed} km/h · lane ${e.lane} · exit in ${Math.max(0, Math.round(1900 - e.position))} m · ticks ${engine.ticks} · ${mode}`;
    tally.textContent =
      `red lights respected ${redRespected}/${redTotal} · pedestrians stopped for ${pedStopped}/${pedTotal} ` +
      `· lane changes chosen ${laneChangeChosen}/${laneChangeTotal}`;
    if (result.ms != null || result.device) ctx.readout(sectionEl, { ms: result.ms, device: result.device });
  }

  async function tick() {
    if (!running) return;
    if (engine.done || engine.candidates().length === 0) {
      restart();
      schedule();
      return;
    }
    const result = mode === 'greedy' ? { move: greedyPolicy(engine), probs: {}, ms: 0 } : await decideMove();
    if (!running) return; // screen may have left the viewport while awaiting decideMove
    if (!result || !result.move) {
      restart();
      schedule();
      return;
    }

    // Tally is about the model's behaviour (demo-spec-v2.md's 0/10 claim), not the scripted
    // greedy policy - only accumulate it while 'model' is the active policy.
    if (mode === 'model') {
      const e = engine.ego;
      const redLight = engine.lights.find((l) => l.pos >= e.position && l.pos - e.position <= RED_LIGHT_RANGE && l.state === 'red');
      if (redLight) {
        redTotal += 1;
        if (result.move === 'brake' || result.move === 'stop') redRespected += 1;
      }
      const pedNear = engine.pedestrians.some((p) => p.lane === e.lane && Math.abs(p.pos - e.position) <= PED_RANGE);
      if (pedNear) {
        pedTotal += 1;
        if (result.move === 'stop') pedStopped += 1;
      }
      laneChangeTotal += 1;
      if (result.move === 'change lane left' || result.move === 'change lane right') laneChangeChosen += 1;
    }

    engine.step(result.move);
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
    entries.some((e) => e.isIntersecting) ? start() : stop();
  }).observe(sectionEl);
}
