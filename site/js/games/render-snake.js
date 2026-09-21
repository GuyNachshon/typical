// Real canvas Snake renderer over js/snake.js's pure engine. 10x10 board, ink cells on putty,
// bone grid gaps, smooth interpolation between ticks, paper death flash.
import { Snake } from '../snake.js';
import { TOKENS, mountChrome, paintDecision, watchVisibility, createTicker, createHumanOverride, bindKeys, modelPolicy, replayFrame, loadJSON, scoreboardLine } from './loop.js';

const TICK_MS = 200;
const KEYMAP = {
  ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
  w: 'up', s: 'down', a: 'left', d: 'right',
};

export async function mount(el, { decide, mode, ctx } = {}) {
  const refs = mountChrome(el, { label: 'Snake · 10×10 board' });
  const canvas = refs.canvas;
  const dctx = canvas.getContext('2d');

  const presets = (await loadJSON('data/presets.json').catch(() => null)) ?? null;
  const question = presets?.snake?.question ?? 'Which move brings the snake closer to the food without dying?';
  // data/replays/snake.json is {frames, summary} (scripts/record_games.mjs) - replayFrame()
  // wants the bare frames array.
  const replay = ((await loadJSON('data/replays/snake.json').catch(() => null)) ?? {}).frames ?? [];

  let engine = new Snake({ w: 10, h: 10 });
  let policyName = mode() === 'live' ? 'model' : 'model'; // MODEL button always present; static plays the replay
  let prevState = engine.state();
  let currState = engine.state();
  let lastTickAt = performance.now();
  let replayIndex = 0;
  let dying = false;
  let deathFlashAt = 0;
  let visible = true;
  let lastDecision = { candidates: engine.safeMoves(), probs: {}, p_null: null, sentence: engine.describe() };
  const human = createHumanOverride(3000);
  let score = { you: 0, model: 0 }; // "you vs model" HUD line — food eaten, reset on restart

  function scriptedMove(e) {
    const safe = e.safeMoves();
    if (safe.length === 0) return safe[0];
    const { head, food } = e.state();
    const D = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
    return safe
      .map((m) => {
        const [dx, dy] = D[m];
        return { m, dist: Math.abs(food.x - (head.x + dx)) + Math.abs(food.y - (head.y + dy)) };
      })
      .sort((a, b) => a.dist - b.dist)[0].m;
  }

  async function tick() {
    // human override takes priority over everything, including the static-mode replay branch
    // below — otherwise arrow keys would silently do nothing whenever policy is "Model" and
    // there's no live server (the common static-hosted case).
    if (human.active() && !engine.dead) {
      const safe = engine.safeMoves();
      if (safe.length === 0) {
        engine.dead = true;
        currState = engine.state();
        flashDeath();
        return;
      }
      const move = safe.includes(human.move) ? human.move : scriptedMove(engine);
      prevState = engine.state();
      engine.step(move);
      currState = engine.state();
      lastTickAt = performance.now();
      lastDecision = { candidates: safe, probs: { [move]: 1 }, p_null: 0, sentence: engine.describe() };
      if (currState.score > prevState.score) score.you++;
      if (engine.dead) flashDeath();
      return;
    }

    if (policyName === 'model' && mode() !== 'live') {
      const f = replayFrame(replay, replayIndex);
      if (!f) return;
      replayIndex = f.nextIndex;
      prevState = currState;
      currState = f.state;
      lastTickAt = performance.now();
      lastDecision = { candidates: f.candidates, probs: f.probs, p_null: f.p_null, sentence: f.desc };
      if (currState.score > prevState.score) score.model++;
      if (currState.dead) flashDeath();
      return;
    }

    if (engine.dead) return;
    const safe = engine.safeMoves();
    if (safe.length === 0) {
      engine.dead = true;
      currState = engine.state();
      flashDeath();
      return;
    }

    let move;
    let decision;
    if (policyName === 'scripted') {
      move = scriptedMove(engine);
      decision = { candidates: safe, probs: { [move]: 1 }, p_null: 0, sentence: engine.describe() };
    } else {
      const r = await modelPolicy({ decide, engine, question: safe.length > 1 ? question : question });
      move = r?.move ?? scriptedMove(engine);
      decision = { candidates: safe, probs: r?.probs ?? {}, p_null: r?.p_null ?? null, sentence: engine.describe() };
    }

    prevState = engine.state();
    engine.step(move);
    currState = engine.state();
    lastTickAt = performance.now();
    lastDecision = decision;
    if (currState.score > prevState.score) score.model++;
    if (engine.dead) flashDeath();
  }

  function flashDeath() {
    dying = true;
    deathFlashAt = performance.now();
    setTimeout(restart, 650);
  }

  function restart() {
    dying = false;
    engine = new Snake({ w: 10, h: 10 });
    prevState = engine.state();
    currState = engine.state();
    replayIndex = 0;
    lastTickAt = performance.now();
    lastDecision = { candidates: engine.safeMoves(), probs: {}, p_null: null, sentence: engine.describe() };
    score = { you: 0, model: 0 };
  }

  function resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = refs.stage.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    canvas.dataset.dpr = dpr;
  }
  const ro = new ResizeObserver(resize);
  ro.observe(refs.stage);
  resize();

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function draw() {
    const dpr = Number(canvas.dataset.dpr || 1);
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    dctx.save();
    dctx.scale(dpr, dpr);
    dctx.clearRect(0, 0, w, h);

    const state = currState;
    const cell = Math.min(w, h) / state.w;
    const boardW = cell * state.w;
    const boardH = cell * state.h;
    const ox = (w - boardW) / 2;
    const oy = (h - boardH) / 2;

    dctx.fillStyle = TOKENS.putty;
    dctx.fillRect(ox, oy, boardW, boardH);

    dctx.strokeStyle = TOKENS.vellum;
    dctx.lineWidth = 1;
    for (let x = 0; x <= state.w; x++) {
      dctx.beginPath();
      dctx.moveTo(ox + x * cell, oy);
      dctx.lineTo(ox + x * cell, oy + boardH);
      dctx.stroke();
    }
    for (let y = 0; y <= state.h; y++) {
      dctx.beginPath();
      dctx.moveTo(ox, oy + y * cell);
      dctx.lineTo(ox + boardW, oy + y * cell);
      dctx.stroke();
    }

    const t = Math.max(0, Math.min(1, (performance.now() - lastTickAt) / TICK_MS));
    const gap = 2;

    // food
    dctx.fillStyle = TOKENS.ink;
    dctx.beginPath();
    const fx = ox + (state.food.x + 0.5) * cell;
    const fy = oy + (state.food.y + 0.5) * cell;
    dctx.arc(fx, fy, cell * 0.28, 0, Math.PI * 2);
    dctx.fill();

    // body segments, tail-first so the head paints on top
    const body = state.body;
    const prevBody = prevState.body;
    for (let i = body.length - 1; i >= 0; i--) {
      const cur = body[i];
      const prev = prevBody[i] ?? cur;
      const bx = ox + (lerp(prev.x, cur.x, t) + 0.5) * cell;
      const by = oy + (lerp(prev.y, cur.y, t) + 0.5) * cell;
      const isHead = i === 0;
      const size = (isHead ? cell * 0.92 : cell * 0.82) - gap;
      dctx.fillStyle = TOKENS.ink;
      dctx.fillRect(bx - size / 2, by - size / 2, size, size);
      if (isHead) {
        dctx.fillStyle = TOKENS.paper;
        dctx.beginPath();
        dctx.arc(bx + size * 0.18, by - size * 0.18, cell * 0.07, 0, Math.PI * 2);
        dctx.fill();
      }
    }

    if (dying) {
      const dt = Math.min(1, (performance.now() - deathFlashAt) / 300);
      dctx.fillStyle = TOKENS.paper;
      dctx.globalAlpha = dt < 0.5 ? dt * 2 : (1 - dt) * 2;
      dctx.fillRect(ox, oy, boardW, boardH);
      dctx.globalAlpha = 1;
    }

    dctx.restore();
  }

  function frame() {
    draw();
    refs.hud.innerHTML = '';
    const l1 = document.createElement('div');
    l1.textContent = `Snake · 10×10 board`;
    const l2 = document.createElement('div');
    l2.textContent = `score ${currState.score} · steps ${currState.steps}`;
    refs.hud.append(l1, l2);
    const scoreLine = scoreboardLine(score);
    if (scoreLine) refs.hud.appendChild(document.createElement('div')).textContent = scoreLine;
    paintDecision(refs, lastDecision);
    rafId = requestAnimationFrame(frame);
  }
  let rafId = requestAnimationFrame(frame);

  const ticker = createTicker(TICK_MS, tick);
  ticker.start();
  const stopWatch = watchVisibility(el, (v) => {
    visible = v;
    if (v) ticker.resume();
    else ticker.pause();
  });
  const unbindKeys = bindKeys(el, KEYMAP, (action) => human.set(action));

  refs.policyBtn.textContent = 'Model';
  refs.policyBtn.classList.add('gc-on');
  refs.policyBtn.addEventListener('click', () => {
    policyName = policyName === 'model' ? 'scripted' : 'model';
    refs.policyBtn.textContent = policyName === 'model' ? 'Model' : 'Scripted';
    refs.policyBtn.classList.toggle('gc-on', policyName === 'model');
  });
  refs.restartBtn.addEventListener('click', restart);

  return {
    stop() {
      cancelAnimationFrame(rafId);
      ticker.stop();
      stopWatch();
      unbindKeys();
    },
    restart,
    setPolicy(name) {
      policyName = name === 'scripted' ? 'scripted' : 'model';
      refs.policyBtn.textContent = policyName === 'model' ? 'Model' : 'Scripted';
      refs.policyBtn.classList.toggle('gc-on', policyName === 'model');
    },
  };
}
