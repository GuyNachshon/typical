import { fmtProb } from './api.js';
// dots.js — the one visual primitive. Every dot's brightness is a probability the model produced
// (or, where no data applies, a fixed value). Canvas-based, retina-aware, no animation on load;
// update() eases alpha only.
//
//   dotText(el, 'TYPICAL', { dot: 20, gap: 8, values })  -> { update(values), cols, rows }
//   dotBars(el, [{label, p, winner, isNull}])            -> { update(rows) }
//   dotField(el, cols, rows, values)                       -> { update(values) }
//
// Colours come from CSS custom properties on :root (--cream, --champagne, --ash) so the palette
// lives in one place.

const FONT = {
  A: ['.###.', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
  B: ['####.', '#...#', '#...#', '####.', '#...#', '#...#', '####.'],
  C: ['.###.', '#...#', '#....', '#....', '#....', '#...#', '.###.'],
  D: ['####.', '#...#', '#...#', '#...#', '#...#', '#...#', '####.'],
  E: ['#####', '#....', '#....', '####.', '#....', '#....', '#####'],
  F: ['#####', '#....', '#....', '####.', '#....', '#....', '#....'],
  G: ['.###.', '#...#', '#....', '#.###', '#...#', '#...#', '.###.'],
  H: ['#...#', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
  I: ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '#####'],
  J: ['..###', '...#.', '...#.', '...#.', '...#.', '#..#.', '.##..'],
  K: ['#...#', '#..#.', '#.#..', '##...', '#.#..', '#..#.', '#...#'],
  L: ['#....', '#....', '#....', '#....', '#....', '#....', '#####'],
  M: ['#...#', '##.##', '#.#.#', '#.#.#', '#...#', '#...#', '#...#'],
  N: ['#...#', '##..#', '#.#.#', '#..##', '#...#', '#...#', '#...#'],
  O: ['.###.', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
  P: ['####.', '#...#', '#...#', '####.', '#....', '#....', '#....'],
  Q: ['.###.', '#...#', '#...#', '#...#', '#.#.#', '#..#.', '.##.#'],
  R: ['####.', '#...#', '#...#', '####.', '#.#..', '#..#.', '#...#'],
  S: ['.####', '#....', '#....', '.###.', '....#', '....#', '####.'],
  T: ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..'],
  U: ['#...#', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
  V: ['#...#', '#...#', '#...#', '#...#', '#...#', '.#.#.', '..#..'],
  W: ['#...#', '#...#', '#...#', '#.#.#', '#.#.#', '##.##', '#...#'],
  X: ['#...#', '#...#', '.#.#.', '..#..', '.#.#.', '#...#', '#...#'],
  Y: ['#...#', '#...#', '.#.#.', '..#..', '..#..', '..#..', '..#..'],
  Z: ['#####', '....#', '...#.', '..#..', '.#...', '#....', '#####'],
  0: ['.###.', '#...#', '#..##', '#.#.#', '##..#', '#...#', '.###.'],
  1: ['..#..', '.##..', '..#..', '..#..', '..#..', '..#..', '.###.'],
  2: ['.###.', '#...#', '....#', '...#.', '..#..', '.#...', '#####'],
  3: ['#####', '...#.', '..#..', '...#.', '....#', '#...#', '.###.'],
  4: ['...#.', '..##.', '.#.#.', '#..#.', '#####', '...#.', '...#.'],
  5: ['#####', '#....', '####.', '....#', '....#', '#...#', '.###.'],
  6: ['..##.', '.#...', '#....', '####.', '#...#', '#...#', '.###.'],
  7: ['#####', '....#', '...#.', '..#..', '.#...', '.#...', '.#...'],
  8: ['.###.', '#...#', '#...#', '.###.', '#...#', '#...#', '.###.'],
  9: ['.###.', '#...#', '#...#', '.####', '....#', '...#.', '.##..'],
  '.': ['.....', '.....', '.....', '.....', '.....', '.##..', '.##..'],
  '%': ['##..#', '##.#.', '...#.', '..#..', '.#...', '#.##.', '#..##'],
  '/': ['....#', '....#', '...#.', '..#..', '.#...', '#....', '#....'],
  '-': ['.....', '.....', '.....', '#####', '.....', '.....', '.....'],
  ':': ['.....', '.##..', '.##..', '.....', '.##..', '.##..', '.....'],
  ' ': ['.....', '.....', '.....', '.....', '.....', '.....', '.....'],
  '∅': ['.###.', '#..##', '#.#.#', '#.#.#', '#.#.#', '##..#', '.###.'],
};

const css = (name, fallback) =>
  (typeof getComputedStyle === 'function' && getComputedStyle(document.documentElement).getPropertyValue(name).trim()) || fallback;

function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

// Layout a string into a boolean grid (cols x 7) with 1 empty column between glyphs.
export function textGrid(text) {
  const glyphs = [...text.toUpperCase()].map((ch) => FONT[ch] || FONT[' ']);
  const cols = glyphs.length * 6 - 1;
  const grid = Array.from({ length: 7 }, () => Array(cols).fill(false));
  glyphs.forEach((g, gi) => {
    for (let r = 0; r < 7; r++) for (let c = 0; c < 5; c++) grid[r][gi * 6 + c] = g[r][c] === '#';
  });
  return grid;
}

// Generic canvas dot grid. `on[r][c]` says whether the dot exists; `values[k]` (0..1) is its
// probability, consumed in reading order over existing dots only.
function makeCanvas(el, cols, rows, dot, gap) {
  const canvas = document.createElement('canvas');
  const w = cols * dot + (cols - 1) * gap;
  const h = rows * dot + (rows - 1) * gap;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = '100%';
  canvas.style.maxWidth = w + 'px';
  canvas.style.height = 'auto';
  canvas.style.display = 'block';
  el.innerHTML = '';
  el.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { canvas, ctx, w, h };
}

function paintGrid(ctx, on, cur, dot, gap, { radius = 0.22 } = {}) {
  const cream = css('--cream', '#fff7dd');
  const champagne = css('--champagne', '#c8ad86');
  const ash = css('--ash', '#66635f');
  ctx.clearRect(0, 0, 1e5, 1e5);
  let k = 0;
  const r = dot * radius;
  for (let row = 0; row < on.length; row++) {
    for (let col = 0; col < on[row].length; col++) {
      if (!on[row][col]) continue;
      const p = cur[k++];
      const x = col * (dot + gap);
      const y = row * (dot + gap);
      // lit dots: champagne at high mass, cream in the middle, ash when the mass is near zero
      let fill;
      // the letterform must always read: an unlit dot is still a dot
      if (p >= 0.5) fill = rgba(champagne, 0.6 + 0.4 * p);
      else if (p >= 0.12) fill = rgba(cream, 0.3 + 0.5 * p);
      else fill = rgba(cream, 0.16 + p);
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.roundRect(x, y, dot, dot, r);
      ctx.fill();
    }
  }
}

function eased(el, on, dot, gap, initial, opts) {
  const { ctx } = makeCanvas(el, on[0].length, on.length, dot, gap);
  const n = on.flat().filter(Boolean).length;
  let cur = Array.from({ length: n }, (_, i) => (initial ? initial[i % initial.length] : 0.9));
  let target = cur.slice();
  let raf = 0;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const draw = () => paintGrid(ctx, on, cur, dot, gap, opts);
  const tick = () => {
    let moving = false;
    for (let i = 0; i < n; i++) {
      const d = target[i] - cur[i];
      if (Math.abs(d) > 0.005) { cur[i] += d * 0.28; moving = true; } else cur[i] = target[i];
    }
    draw();
    raf = moving ? requestAnimationFrame(tick) : 0;
  };
  draw();
  return {
    update(values) {
      if (!values || !values.length) return;
      target = Array.from({ length: n }, (_, i) => values[i % values.length]);
      if (reduce) { cur = target.slice(); draw(); return; }
      if (!raf) raf = requestAnimationFrame(tick);
    },
    // feed a stream: push one probability vector, it scrolls into the field from the end
    push(vec) {
      target = target.slice(vec.length).concat(vec.slice(0, n));
      while (target.length < n) target.push(0.05);
      if (!raf) raf = requestAnimationFrame(tick);
    },
    n,
  };
}

export function dotText(el, text, { dot = 16, gap = 6, values } = {}) {
  return eased(el, textGrid(text), dot, gap, values);
}

export function dotField(el, cols, rows, { dot = 10, gap = 4, values } = {}) {
  const on = Array.from({ length: rows }, () => Array(cols).fill(true));
  return eased(el, on, dot, gap, values);
}

// dotBars: DOM rows (label | 24-dot strip | numeral). Winner dots are champagne, others cream,
// ∅ is drawn with hollow dots.
export function dotBars(el, rows, { dots = 24 } = {}) {
  el.innerHTML = '';
  el.classList.add('dotbars');
  const render = (rs) => {
    el.innerHTML = '';
    const max = Math.max(...rs.map((r) => r.p));
    rs.forEach((r) => {
      const row = document.createElement('div');
      row.className = 'dotbars-row' + (r.isNull ? ' is-null' : '') + (r.p === max && !r.isNull ? ' is-winner' : '');
      const lit = Math.round(r.p * dots);
      const strip = document.createElement('span');
      strip.className = 'dotbars-strip';
      for (let i = 0; i < dots; i++) {
        const d = document.createElement('i');
        if (i < lit) d.className = 'on';
        strip.appendChild(d);
      }
      const label = document.createElement('span');
      label.className = 'dotbars-label';
      label.textContent = r.label; // labels can be user-typed (Try it): text only
      const num = document.createElement('span');
      num.className = 'dotbars-num';
      num.textContent = fmtProb(r.p);
      row.append(label, strip, num);
      el.appendChild(row);
    });
  };
  render(rows);
  return { update: render };
}

// Self-test (node): font coverage and grid geometry.
function selfTest() {
  const g = textGrid('TYPICAL');
  console.assert(g.length === 7 && g[0].length === 7 * 6 - 1, 'TYPICAL grid is 7 x 41');
  console.assert(g[0][0] && g[0][4] && g[6][2] && !g[6][0], 'T glyph shape');
  for (const ch of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.%/-: ∅') {
    const f = FONT[ch];
    console.assert(f && f.length === 7 && f.every((r) => r.length === 5), `glyph ${ch} is 5x7`);
  }
  console.log('dots.js self-test OK');
}
if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) selfTest();
export { selfTest, FONT };
