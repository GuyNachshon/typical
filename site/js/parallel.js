// parallel.js -- the same work, laid out two ways.
//
// Every sentence on this page about "one pass" is asking the reader to picture something, so this
// draws it. Two identical grids of the same forty cells. The top one fills a cell at a time,
// because that is the only order an autoregressive model can produce answers in: the next one is
// conditioned on the last. The bottom one fills at once, because a decision model reads the state
// into a cache and scores every question's suffix against it in the same forward pass.
//
// The figure is SCHEMATIC and carries no numbers on purpose. The measured version of exactly this
// contrast is the pair of terminals below it, running at recorded time. A diagram that invented
// its own timings next to a figure that measured them would undercut the one that did the work.
//
//   mountParallel(container)
//
// Monochrome, like everything else here. The contrast is carried by fill order, which is the thing
// the figure is about -- colour would only be decoration competing with the grids.

import { svgEl, svgText, registerChart, fitWidth, INK, MID, STEEL } from './charts.js';

const COLS = 8;
const ROWS = 5;
const CELL = 22;
const GAP = 4;
const STACK_W = 660; // below this the two grids stack instead of sitting side by side

export const N = COLS * ROWS;
const gridW = COLS * CELL + (COLS - 1) * GAP;
const gridH = ROWS * CELL + (ROWS - 1) * GAP;

// Raster order, which is the order a reader's eye already expects to be filled in.
export function cellAt(i) {
  return { x: (i % COLS) * (CELL + GAP), y: Math.floor(i / COLS) * (CELL + GAP) };
}

// How many cells are lit at time t. The serial lane spends the whole span; the parallel lane spends
// one slot and then holds, which is the entire claim of the figure.
export function litAt(t, span, serial) {
  if (!serial) return t >= span / N ? N : 0;
  return Math.max(0, Math.min(N, Math.floor((t / span) * N)));
}

function label(x, y, str, attrs = {}) {
  return svgText(x, y, str, { 'font-size': 11, 'letter-spacing': 0.72, fill: MID, ...attrs });
}

function grid(x, y) {
  const g = svgEl('g', { transform: `translate(${x},${y})` });
  const cells = [];
  for (let i = 0; i < N; i += 1) {
    const { x: cx, y: cy } = cellAt(i);
    const r = svgEl('rect', { x: cx, y: cy, width: CELL, height: CELL, fill: STEEL, opacity: 0.45 });
    g.appendChild(r);
    cells.push(r);
  }
  return { g, cells };
}

function lane(x, y, w, tag, note, serial) {
  const g = svgEl('g', { transform: `translate(${x},${y})` });
  g.appendChild(label(0, 0, tag.toUpperCase(), { fill: INK, 'font-weight': 700 }));
  const { g: gg, cells } = grid(0, 18);
  g.appendChild(gg);
  const foot = svgText(0, 18 + gridH + 22, '', { 'font-size': 11, fill: MID });
  g.appendChild(foot);
  g.appendChild(label(gridW + 26, 18 + gridH / 2 + 4, note, { fill: MID }));
  return { g, cells, foot, serial };
}

const DESIGN_W = 1080;

export function mountParallel(container) {
  if (!container) return;
  let raf = 0;
  let io = null;
  // registerChart only fires on resize, so the first paint is ours to make
  const draw = () => {
    cancelAnimationFrame(raf);
    if (io) io.disconnect();
    const host = container;
    const width = fitWidth(container, DESIGN_W);
    container.style.maxWidth = `${DESIGN_W}px`;
    const stacked = width < STACK_W;
    const laneW = stacked ? width : (width - 60) / 2;
    const h = stacked ? 2 * (gridH + 70) + 20 : gridH + 78;
    const svg = svgEl('svg', { viewBox: `0 0 ${width} ${h}`, width, height: h, role: 'img' });
    svg.appendChild(svgEl('title', {})).textContent =
      'Two identical grids of forty cells: the autoregressive lane fills one cell at a time, the decision-model lane fills all forty at once.';

    const a = lane(0, 16, laneW, 'one token at a time', 'each answer waits for the last', true);
    const b = lane(stacked ? 0 : laneW + 60, stacked ? gridH + 86 : 16, laneW, 'one pass', 'every question reads the same cache', false);
    svg.append(a.g, b.g);
    host.replaceChildren(svg);

    const span = 4200;
    let t0 = 0;
    const paint = (lit, ln) => {
      ln.cells.forEach((c, i) => {
        c.setAttribute('fill', i < lit ? INK : STEEL);
        c.setAttribute('opacity', i < lit ? 1 : 0.45);
      });
      ln.foot.textContent = `${lit} of ${N}`;
    };
    const step = (now) => {
      const t = now - t0;
      paint(litAt(t, span, true), a);
      paint(litAt(t, span, false), b);
      if (t < span) raf = requestAnimationFrame(step);
    };
    const run = () => {
      cancelAnimationFrame(raf);
      if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
        paint(N, a);
        paint(N, b);
        return;
      }
      t0 = performance.now();
      raf = requestAnimationFrame(step);
    };
    io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) run(); }, { threshold: 0.4 });
    io.observe(host);
    run();
  };
  draw();
  registerChart(container, draw);
}

export function selfTest() {
  console.assert(N === 40, 'forty cells, both lanes');
  console.assert(cellAt(0).x === 0 && cellAt(COLS).y === CELL + GAP, 'raster order wraps at the column count');
  console.assert(litAt(0, 1000, true) === 0 && litAt(1000, 1000, true) === N, 'the serial lane spends the whole span');
  console.assert(litAt(500, 1000, true) === N / 2, 'and is linear in between');
  console.assert(litAt(0, 1000, false) === 0, 'the parallel lane is not lit before it has run');
  console.assert(litAt(1000 / N, 1000, false) === N, 'and lands whole, in one slot');
  console.assert(litAt(500, 1000, false) === N, 'then holds');
  console.log('parallel.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) selfTest();
