// Real canvas Snake renderer over js/snake.js's pure engine. 10x10 board drawn as a green
// phosphor terminal: a monospace glyph grid inside a box-drawn frame, scanlines + a cheap
// glyph bloom (crt.js), ~2-tick phosphor persistence on cells that just went dark, and a
// terminal status line baked into the same canvas so the panel reads as one machine.
import { Snake, greedyPolicy } from '../snake.js';
import { TOKENS, mountChrome, paintDecision, watchVisibility, createTicker, createHumanOverride, bindKeys, modelPolicy, replayFrame, loadJSON, scoreboardLine } from './loop.js';
import { drawScanlines, drawGlyphBloom, phosphorDecay, prefersReducedMotion } from './crt.js';

const TICK_MS = 200;
const FADE_MS = TICK_MS * 2; // phosphor persistence window
const KEYMAP = {
  ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
  w: 'up', s: 'down', a: 'left', d: 'right',
};

// Green P1 phosphor (VT100/Apple II/IBM 5151 terminals) rather than amber: it's the more
// common "terminal" association, and a single hue read against near-black lets brightness
// alone carry head/body/food/wall hierarchy - the monochrome-CRT way, no extra hues to manage.
const P = [77, 255, 136];
const phos = (a) => `rgba(${P[0]},${P[1]},${P[2]},${a})`;
const HEAD_GLYPH = { up: '^', down: 'v', left: '<', right: '>' };
const BODY_GLYPH = 'o';
const FOOD_GLYPH = '*';
const DOT_GLYPH = '·';
const BORDER = { tl: '┌', tr: '┐', bl: '└', br: '┘', h: '─', v: '│' };

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

  const scriptedMove = greedyPolicy; // the rule list applied literally (snake.js RULES)

  // Phosphor persistence: `lastLit` is what was on as of the most recently *committed* tick;
  // `fading` holds cells that just went dark, keyed "x,y", so draw() can afterglow them for
  // FADE_MS without re-deriving a diff every frame. Both mutated in place - no per-frame
  // allocation in the draw loop itself (commit only runs once per tick, gated below).
  let lastLit = new Map();
  const fading = new Map();
  let lastPhosphorTick = -1;
  const reducedMotion = prefersReducedMotion();

  function litCellsOf(state) {
    const m = new Map();
    state.body.forEach((seg, i) => {
      m.set(`${seg.x},${seg.y}`, { ch: i === 0 ? HEAD_GLYPH[state.dir] : BODY_GLYPH, hot: i === 0 });
    });
    m.set(`${state.food.x},${state.food.y}`, { ch: FOOD_GLYPH, hot: true });
    return m;
  }

  function commitPhosphorTick(state, now) {
    const lit = litCellsOf(state);
    for (const [key, info] of lastLit) {
      if (!lit.has(key)) fading.set(key, { ...info, offAt: now });
    }
    for (const key of lit.keys()) fading.delete(key);
    lastLit = lit;
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
    fading.clear();
    lastLit = new Map();
    lastPhosphorTick = -1;
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

  // Font strings are rebuilt only when the cell size actually changes (resize), not every
  // frame - the draw loop below just swaps between these cached strings.
  let cachedCell = 0;
  let font = '';
  let haloFont = '';
  let statusFont = '';

  function draw() {
    const dpr = Number(canvas.dataset.dpr || 1);
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    dctx.save();
    dctx.scale(dpr, dpr);
    dctx.clearRect(0, 0, w, h);

    const state = currState;
    const now = performance.now();

    // Layout: a 1-cell box-drawn frame around the w×h play field, plus one more cell of
    // height below it for the terminal status line - all three read as one boxed screen.
    const outerCols = state.w + 2;
    const outerRows = state.h + 2;
    const cell = Math.min(w / outerCols, h / (outerRows + 1));
    const boardW = cell * outerCols;
    const boardH = cell * outerRows;
    const ox = (w - boardW) / 2;
    const oy = (h - (boardH + cell)) / 2;
    const gx = ox + cell; // interior (play field) origin, inside the frame
    const gy = oy + cell;

    if (cell !== cachedCell) {
      cachedCell = cell;
      const family = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      font = `${Math.round(cell * 0.82)}px ${family}`;
      haloFont = `${Math.round(cell * 1.18)}px ${family}`;
      statusFont = `${Math.round(cell * 0.5)}px ${family}`;
    }

    // CRT glass - same near-black TOKENS.putty the other game cards sit on.
    dctx.fillStyle = TOKENS.putty;
    dctx.fillRect(ox, oy, boardW, boardH + cell);

    dctx.textAlign = 'center';
    dctx.textBaseline = 'middle';
    dctx.font = font;

    // border ring - the "wall" glyph, box-drawing characters at mid brightness so the living
    // snake still reads as the brightest thing on screen
    dctx.fillStyle = phos(0.55);
    for (let x = 0; x < outerCols; x++) {
      const cx = ox + (x + 0.5) * cell;
      const top = x === 0 ? BORDER.tl : x === outerCols - 1 ? BORDER.tr : BORDER.h;
      const bot = x === 0 ? BORDER.bl : x === outerCols - 1 ? BORDER.br : BORDER.h;
      dctx.fillText(top, cx, oy + 0.5 * cell);
      dctx.fillText(bot, cx, oy + (outerRows - 0.5) * cell);
    }
    for (let y = 1; y < outerRows - 1; y++) {
      const cy = oy + (y + 0.5) * cell;
      dctx.fillText(BORDER.v, ox + 0.5 * cell, cy);
      dctx.fillText(BORDER.v, ox + (outerCols - 0.5) * cell, cy);
    }

    // commit the tick→tick diff once per tick (not per frame) so a cell that just went dark
    // starts fading here rather than being recomputed every rAF
    if (lastTickAt !== lastPhosphorTick) {
      commitPhosphorTick(state, lastTickAt);
      lastPhosphorTick = lastTickAt;
    }
    for (const [key, info] of fading) {
      if (reducedMotion || now - info.offAt > FADE_MS) fading.delete(key);
    }

    // play field: lit (head/body/food) > fading (phosphor afterglow) > dim dot-matrix rest
    for (let x = 0; x < state.w; x++) {
      for (let y = 0; y < state.h; y++) {
        const key = `${x},${y}`;
        const cx = gx + (x + 0.5) * cell;
        const cy = gy + (y + 0.5) * cell;
        const lit = lastLit.get(key);
        if (lit) {
          const color = phos(lit.hot ? 1 : 0.85);
          if (lit.hot) {
            drawGlyphBloom(dctx, lit.ch, cx, cy, { font, haloFont, color, haloColor: phos(0.28) });
          } else {
            dctx.font = font;
            dctx.fillStyle = color;
            dctx.fillText(lit.ch, cx, cy);
          }
          continue;
        }
        const fade = fading.get(key);
        if (fade) {
          const a = phosphorDecay(now - fade.offAt, FADE_MS);
          dctx.font = font;
          dctx.fillStyle = phos((fade.hot ? 0.9 : 0.6) * a);
          dctx.fillText(fade.ch, cx, cy);
          continue;
        }
        dctx.font = font;
        dctx.fillStyle = phos(0.12);
        dctx.fillText(DOT_GLYPH, cx, cy);
      }
    }

    if (dying) {
      const dt = Math.min(1, (now - deathFlashAt) / 300);
      dctx.fillStyle = TOKENS.paper;
      dctx.globalAlpha = dt < 0.5 ? dt * 2 : (1 - dt) * 2;
      dctx.fillRect(ox, oy, boardW, boardH + cell);
      dctx.globalAlpha = 1;
    }

    // terminal status line - the same phosphor, inside the frame's own boxed screen
    dctx.textAlign = 'left';
    dctx.textBaseline = 'middle';
    dctx.font = statusFont;
    dctx.fillStyle = phos(0.75);
    const pad = (n, digits) => String(n).padStart(digits, '0');
    const cursor = reducedMotion || Math.floor(now / 500) % 2 === 0 ? '█' : ' ';
    dctx.fillText(
      `SCORE ${pad(state.score, 3)}  LEN ${pad(state.body.length, 2)}  TICK ${pad(state.steps, 4)} ${cursor}`,
      ox + cell * 0.3,
      oy + boardH + cell / 2
    );

    drawScanlines(dctx, ox, oy, boardW, boardH + cell);
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
