// Real canvas Tetris renderer over js/games/tetris.js's pure engine. Same green-phosphor
// terminal treatment as render-snake.js (crt.js: scanlines, glyph bloom, phosphor persistence) -
// the well is a monospace glyph grid inside a box-drawn frame, so it reads as a sibling of the
// Snake card rather than a different game entirely.
//
// One tick = one piece: the engine doesn't step gravity frame by frame, it resolves a whole
// hard-drop placement per decision (site/data/demos/spec-v4.md's point - the model picks the
// placement, not the keystrokes). currState.current is always the *next* piece waiting on its
// decision, so each redraw shows the settled stack plus that piece hovering at the top; next
// tick it either has become part of the stack (a new piece landed) or, on the fatal spawn, the
// board flashes and resets.
import { Tetris, greedyPolicy, QUESTION } from './tetris.js';
import { TOKENS, mountChrome, paintDecision, watchVisibility, createTicker, createHumanOverride, bindKeys, modelPolicy, replayFrame, loadJSON, scoreboardLine } from './loop.js';
import { drawScanlines, drawGlyphBloom, phosphorDecay, prefersReducedMotion } from './crt.js';

const TICK_MS = 450;
const FADE_MS = TICK_MS * 2; // phosphor persistence window, same ratio as Snake
const BOARD = { w: 8, h: 14 };

// Number keys pick a candidate by offered index directly; arrows double up onto the first four
// slots so loop.js's fixed "arrows to take over" hint still does something sensible without
// editing loop.js (candidate labels are dynamic per piece, unlike Snake's fixed up/down/left/
// right, so a static KEYMAP can't name them - tick() below resolves the index against whatever
// candidates() returns that tick).
const KEYMAP = {
  ArrowUp: '0', ArrowDown: '1', ArrowLeft: '2', ArrowRight: '3',
  1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5',
};

// Green P1 phosphor, same palette as render-snake.js - one hue, brightness carries hierarchy.
const P = [77, 255, 136];
const phos = (a) => `rgba(${P[0]},${P[1]},${P[2]},${a})`;
// Distinct glyph per piece type (monochrome, so shape - not hue - is what varies) for both the
// settled stack and the falling piece; falling piece gets the bloom treatment, settled cells don't.
const TYPE_GLYPH = { I: '█', O: '▣', T: '▲', S: '▞', Z: '▚', J: '◣', L: '◢' };
const DOT_GLYPH = '·';
const BORDER = { tl: '┌', tr: '┐', bl: '└', br: '┘', h: '─', v: '│' };
const pad = (n, digits) => String(n).padStart(digits, '0');

export async function mount(el, { decide, mode, ctx } = {}) {
  const refs = mountChrome(el, { label: `Tetris · ${BOARD.w}×${BOARD.h} well` });
  const canvas = refs.canvas;
  const dctx = canvas.getContext('2d');

  const presets = (await loadJSON('data/presets.json').catch(() => null)) ?? null;
  const question = presets?.tetris?.question ?? QUESTION;
  const replay = ((await loadJSON('data/replays/tetris.json').catch(() => null)) ?? {}).frames ?? [];

  let engine = new Tetris(BOARD);
  let policyName = 'model'; // MODEL button always present; static plays the replay
  let prevState = engine.state();
  let currState = engine.state();
  let lastTickAt = performance.now();
  let replayIndex = 0;
  let dying = false;
  let deathFlashAt = 0;
  let visible = true;
  let lastDecision = { candidates: engine.candidates(), probs: {}, p_null: null, sentence: engine.describe() };
  const human = createHumanOverride(3000);
  let score = { you: 0, model: 0 }; // "you vs model" HUD line - lines cleared, reset on restart

  const scriptedMove = greedyPolicy; // the rule list applied literally (tetris.js RULES)

  // Phosphor persistence, identical pattern to render-snake.js: `lastLit` is what was on as of
  // the most recently committed tick; `fading` holds cells that just went dark (a locked piece's
  // spawn-preview glyphs, or a cleared line) so draw() can afterglow them without a per-frame diff.
  let lastLit = new Map();
  const fading = new Map();
  let lastPhosphorTick = -1;
  const reducedMotion = prefersReducedMotion();

  function litCellsOf(state) {
    const m = new Map();
    for (let y = 0; y < state.h; y++) {
      for (let x = 0; x < state.w; x++) {
        const c = state.board[y][x];
        if (c) m.set(`${x},${y}`, { ch: TYPE_GLYPH[c] ?? '█', hot: false });
      }
    }
    if (state.current) {
      const glyph = TYPE_GLYPH[state.current.type] ?? '█';
      state.current.cells.forEach(([dx, dy]) => {
        m.set(`${state.current.col + dx},${dy}`, { ch: glyph, hot: true });
      });
    }
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
    if (human.active() && !engine.dead) {
      const cands = engine.candidates();
      if (cands.length === 0) {
        engine.dead = true;
        currState = engine.state();
        flashDeath();
        return;
      }
      const idx = Number(human.move);
      const move = cands[idx] ?? scriptedMove(engine);
      prevState = engine.state();
      engine.step(move);
      currState = engine.state();
      lastTickAt = performance.now();
      lastDecision = { candidates: cands, probs: { [move]: 1 }, p_null: 0, sentence: engine.describe() };
      if (currState.lines > prevState.lines) score.you++;
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
      if (currState.lines > prevState.lines) score.model++;
      if (currState.dead) flashDeath();
      return;
    }

    if (engine.dead) return;
    const cands = engine.candidates();
    if (cands.length === 0) {
      engine.dead = true;
      currState = engine.state();
      flashDeath();
      return;
    }

    let move;
    let decision;
    if (policyName === 'scripted') {
      move = scriptedMove(engine);
      decision = { candidates: cands, probs: { [move]: 1 }, p_null: 0, sentence: engine.describe() };
    } else {
      const r = await modelPolicy({ decide, engine, question });
      move = r?.move ?? scriptedMove(engine);
      decision = { candidates: cands, probs: r?.probs ?? {}, p_null: r?.p_null ?? null, sentence: engine.describe() };
    }

    prevState = engine.state();
    engine.step(move);
    currState = engine.state();
    lastTickAt = performance.now();
    lastDecision = decision;
    if (currState.lines > prevState.lines) score.model++;
    if (engine.dead) flashDeath();
  }

  function flashDeath() {
    dying = true;
    deathFlashAt = performance.now();
    setTimeout(restart, 650);
  }

  function restart() {
    dying = false;
    engine = new Tetris(BOARD);
    prevState = engine.state();
    currState = engine.state();
    replayIndex = 0;
    lastTickAt = performance.now();
    lastDecision = { candidates: engine.candidates(), probs: {}, p_null: null, sentence: engine.describe() };
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

    // Same layout solve as render-snake.js: the box (border included) sits clear of the
    // top-right HUD and the bottom-left decision/controls chrome loop.js overlays, by standing
    // beside the decision readout when there's room and above it otherwise.
    const outerCols = state.w + 2;
    const outerRows = state.h + 2;
    const topClear = 90;
    const sideMargin = 10;
    const footW = Math.min(520, w * 0.58);
    const footH = 150;
    const rightRoom = w - footW - sideMargin * 2;
    const beside = rightRoom > (h - topClear - sideMargin) * 0.62;
    const availW = beside ? rightRoom : w - sideMargin * 2;
    const availH = beside ? h - topClear - sideMargin : h - topClear - footH - sideMargin;
    const cell = Math.max(6, Math.min(availW / outerCols, availH / outerRows));
    const boardW = cell * outerCols;
    const boardH = cell * outerRows;
    const ox = beside ? footW + sideMargin + (rightRoom - boardW) / 2 : (w - boardW) / 2;
    const oy = topClear + Math.max(0, availH - boardH) / 2;
    const gx = ox + cell;
    const gy = oy + cell;

    if (cell !== cachedCell) {
      cachedCell = cell;
      const family = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      font = `${Math.round(cell * 0.82)}px ${family}`;
      haloFont = `${Math.round(cell * 1.18)}px ${family}`;
      statusFont = `${Math.round(cell * 0.46)}px ${family}`;
    }

    dctx.fillStyle = TOKENS.putty;
    dctx.fillRect(ox, oy, boardW, boardH);

    dctx.textAlign = 'center';
    dctx.textBaseline = 'middle';
    dctx.font = font;

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

    // terminal status line set into the top border, same trick as render-snake.js's title bar -
    // the one spot neither the HUD nor the decision chrome ever covers.
    dctx.font = statusFont;
    dctx.textAlign = 'left';
    const titleAvail = boardW - 3 * cell;
    let title = `LINES ${pad(state.lines, 3)}  PIECES ${pad(state.pieces, 3)}`;
    if (dctx.measureText(title).width > titleAvail) title = `L${pad(state.lines, 3)} P${pad(state.pieces, 3)}`;
    if (dctx.measureText(title).width > titleAvail) title = '';
    if (title) {
      const titleX = ox + 1.5 * cell;
      const titleW = dctx.measureText(title).width;
      dctx.fillStyle = TOKENS.putty;
      dctx.fillRect(titleX - cell * 0.25, oy + cell * 0.12, titleW + cell * 0.5, cell * 0.76);
      dctx.fillStyle = phos(0.75);
      dctx.textBaseline = 'middle';
      dctx.fillText(title, titleX, oy + 0.5 * cell);
    }
    dctx.textAlign = 'center';
    dctx.font = font;

    if (lastTickAt !== lastPhosphorTick) {
      commitPhosphorTick(state, lastTickAt);
      lastPhosphorTick = lastTickAt;
    }
    for (const [key, info] of fading) {
      if (reducedMotion || now - info.offAt > FADE_MS) fading.delete(key);
    }

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
      dctx.fillRect(ox, oy, boardW, boardH);
      dctx.globalAlpha = 1;
    }

    drawScanlines(dctx, ox, oy, boardW, boardH);
    dctx.restore();
  }

  function frame() {
    draw();
    refs.hud.innerHTML = '';
    const l1 = document.createElement('div');
    l1.textContent = `Tetris · ${BOARD.w}×${BOARD.h} well`;
    const l2 = document.createElement('div');
    l2.textContent = `lines ${currState.lines} · pieces ${currState.pieces}`;
    refs.hud.append(l1, l2);
    const scoreLine = scoreboardLine(score, 'lines');
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
