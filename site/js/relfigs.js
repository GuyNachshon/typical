// relfigs.js -- the two arguments on the launch page that were carrying no figure at all.
//
// Both sections were a wall of prose next to a lot of empty paper. Neither needed a new number:
// the evidence was already in the report and in the model cards, it just had never been drawn.
//
//   mountDeltaQ(container)  the control that killed our first design, and what replaced it
//   mountTypes(container)   the four probability spaces, as shapes
//
// mountTypes is schematic and says so in its caption. mountDeltaQ is not: every value in it is
// measured and cited below.

import { svgEl, svgText, registerChart, fitWidth, scale, INK, MID, STEEL } from './charts.js';

const DESIGN_W = 1080;

// Δ_q is the accuracy a model loses when the real question is replaced by a shuffled one. A design
// that scores the same either way has not used the question -- it has learned which answers look
// plausible in general. That is exactly what our candidate-blind state did, at full depth, across
// every student shape we tried: −0.008, within noise of zero (research.md §c, REPORT §3s). Putting
// the options inside the decision pass is what the other two rows are.
export const DELTA_Q = [
  { label: 'candidate-blind state', note: 'abandoned · 28 layers', v: -0.008, ours: false },
  { label: 'typical-small', note: 'shipped', v: 0.127, ours: true },
  { label: 'typical-medium', note: 'shipped', v: 0.193, ours: true },
];

const fmt = (v) => `${v < 0 ? '−' : '+'}${Math.abs(v).toFixed(3).replace(/^0/, '')}`;

export function mountDeltaQ(container) {
  if (!container) return;
  const draw = () => {
    const w = fitWidth(container, DESIGN_W);
    const labelW = Math.min(240, Math.max(150, w * 0.24));
    const pad = { t: 18, r: 78, b: 40 };
    const rowH = 54;
    const h = pad.t + DELTA_Q.length * rowH + pad.b;
    const iw = w - labelW - pad.r;
    container.replaceChildren();
    container.style.maxWidth = `${DESIGN_W}px`;
    const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', role: 'img' });
    const t = svgEl('title');
    t.textContent = 'The shuffled-question control: the abandoned candidate-blind design scores the same with the question replaced, the shipped models do not.';
    svg.appendChild(t);

    // the domain is pinned around zero because zero is the claim: a bar that does not cross it is
    // a model that did not read the question
    const x = scale([-0.03, 0.22], [0, iw]);
    const g = svgEl('g', { transform: `translate(${labelW},${pad.t})` });
    svg.appendChild(g);
    const zero = x(0);

    DELTA_Q.forEach((r, i) => {
      const y = i * rowH;
      g.appendChild(svgText(-labelW, y + 18, r.label, { 'font-size': 13, fill: r.ours ? INK : MID }));
      g.appendChild(svgText(-labelW, y + 34, r.note, { 'font-size': 11, fill: STEEL, 'letter-spacing': 0.36 }));
      const bw = Math.abs(x(r.v) - zero);
      g.appendChild(svgEl('rect', {
        x: r.v < 0 ? zero - bw : zero, y: y + 8, width: Math.max(bw, 1), height: 20,
        fill: r.ours ? INK : MID, opacity: r.ours ? 1 : 0.5,
      }));
      g.appendChild(svgText(x(r.v) + (r.v < 0 ? -8 : 8), y + 23, fmt(r.v), {
        'font-size': 12, fill: r.ours ? INK : MID, 'text-anchor': r.v < 0 ? 'end' : 'start',
      }));
    });

    g.appendChild(svgEl('line', { x1: zero, y1: -6, x2: zero, y2: DELTA_Q.length * rowH - 4, stroke: INK, 'stroke-width': 1 }));
    g.appendChild(svgText(zero, DELTA_Q.length * rowH + 14, '0 · the question made no difference', { 'font-size': 11, fill: MID, 'text-anchor': 'middle' }));
    container.appendChild(svg);
  };
  draw();
  registerChart(container, draw);
}

// The four primitives, drawn as the probability space each one owns. No values: these are shapes,
// and the numbers that belong to each type are in the prose beside them.
const TYPES = [
  { name: 'choice', bars: [0.12, 0.62, 0.18, 0.08], note: 'one of K, defined at call time' },
  { name: 'noul', bars: [0.16, 0.84], note: 'one Bernoulli, from a head that never sees the labels' },
  { name: 'score', bars: [0.03, 0.11, 0.26, 0.6], note: 'ordered levels, mass near the true one', ordered: true },
  { name: 'abstain', bars: [0.58, 0.2, 0.09], tail: 0.13, note: 'p(none of these), on every Choice and Score' },
];

export function mountTypes(container) {
  if (!container) return;
  const draw = () => {
    const w = fitWidth(container, DESIGN_W);
    const cols = w < 620 ? 2 : 4;
    const colW = (w - (cols - 1) * 20) / cols;
    const rows = Math.ceil(TYPES.length / cols);
    const plotH = 84;
    const cardH = plotH + 74;
    const h = rows * cardH + (rows - 1) * 16;
    container.replaceChildren();
    container.style.maxWidth = `${DESIGN_W}px`;
    const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', role: 'img' });
    const t = svgEl('title');
    t.textContent = 'Four probability spaces drawn as shapes: a categorical over K, a single Bernoulli, an ordered ladder, and an abstention set apart from the options.';
    svg.appendChild(t);

    TYPES.forEach((ty, i) => {
      const gx = (i % cols) * (colW + 20);
      const gy = Math.floor(i / cols) * (cardH + 16);
      const g = svgEl('g', { transform: `translate(${gx},${gy})` });
      g.appendChild(svgEl('line', { x1: 0, y1: 0, x2: colW, y2: 0, stroke: STEEL, 'stroke-width': 1 }));
      g.appendChild(svgText(0, 20, ty.name.toUpperCase(), { 'font-size': 11, 'font-weight': 700, 'letter-spacing': 0.72, fill: INK }));

      const n = ty.bars.length + (ty.tail != null ? 1 : 0);
      const gap = 8;
      const bw = (colW - (n - 1) * gap) / n;
      const top = 38;
      ty.bars.forEach((v, j) => {
        const bh = Math.max(2, v * plotH);
        g.appendChild(svgEl('rect', { x: j * (bw + gap), y: top + plotH - bh, width: bw, height: bh, fill: INK, opacity: ty.ordered ? 0.35 + 0.65 * v : (v === Math.max(...ty.bars) ? 1 : 0.32) }));
      });
      if (ty.tail != null) {
        const j = ty.bars.length;
        const bh = Math.max(2, ty.tail * plotH);
        // the abstention is not one of your options, so it does not share their baseline
        g.appendChild(svgEl('rect', { x: j * (bw + gap), y: top + plotH - bh, width: bw, height: bh, fill: STEEL }));
        g.appendChild(svgEl('line', { x1: j * (bw + gap) - gap / 2, y1: top, x2: j * (bw + gap) - gap / 2, y2: top + plotH, stroke: STEEL, 'stroke-dasharray': '2 3' }));
      }
      g.appendChild(svgEl('line', { x1: 0, y1: top + plotH + 1, x2: colW, y2: top + plotH + 1, stroke: STEEL }));
      const note = svgText(0, top + plotH + 22, '', { 'font-size': 11, fill: MID });
      // one wrap point is enough at these widths, and it keeps the four cards the same height
      const words = ty.note.split(' ');
      let line = '';
      const lines = [];
      words.forEach((word) => {
        if ((line + ' ' + word).trim().length * 6.3 > colW && line) { lines.push(line); line = word; } else { line = (line + ' ' + word).trim(); }
      });
      lines.push(line);
      lines.slice(0, 2).forEach((ln, k) => {
        const ts = svgEl('tspan', { x: 0, dy: k === 0 ? 0 : 14 });
        ts.textContent = ln;
        note.appendChild(ts);
      });
      g.appendChild(note);
      svg.appendChild(g);
    });
    container.appendChild(svg);
  };
  draw();
  registerChart(container, draw);
}

export function selfTest() {
  console.assert(DELTA_Q[0].v < 0 && !DELTA_Q[0].ours, 'the abandoned design is the one that failed the control');
  console.assert(DELTA_Q[1].v === 0.127 && DELTA_Q[2].v === 0.193, 'the shipped values are the model cards');
  console.assert(fmt(-0.008) === '−.008' && fmt(0.127) === '+.127', 'a signed value keeps its sign and drops its leading zero');
  const s = scale([-0.03, 0.22], [0, 100]);
  console.assert(s(0) > 0 && s(0) < 100, 'zero is inside the frame, which is the whole point of the frame');
  console.assert(TYPES.length === 4 && TYPES.filter((t) => t.tail != null).length === 1, 'only abstain draws a tail');
  console.assert(Math.max(...TYPES[2].bars) === TYPES[2].bars[3] && TYPES[2].ordered, 'score peaks at its top level and is drawn as a ladder');
  console.log('relfigs.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) selfTest();
