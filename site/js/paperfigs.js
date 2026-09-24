// paperfigs.js -- the figures the research page is built around.
//
// Every value comes from site/data/paper-*.json, which is transcribed digit-for-digit from the paper
// on branch main. Nothing here computes a result; these draw ones that were already measured.
//
//   mountDeltaQ(host, doc)    the shuffled-question control -- the page's centrepiece
//   mountReadouts(host, doc)  three ways to read an answer out of one pass
//   mountLadder(host, doc)    JevBench against backbone size, trained and frozen

import { svgEl, svgText, registerChart, fitWidth, scale, logScale, placeLabels, INK, MID, STEEL, OURS, OTHER, FONT_MONO } from './charts.js';

const W = 1080;
const num = (v, d = 2) => (v == null ? '—' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(d).replace(/^0/, ''));

function frame(host, w, h, title) {
  host.replaceChildren();
  host.style.maxWidth = `${W}px`;
  const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', role: 'img' });
  const t = svgEl('title');
  t.textContent = title;
  svg.appendChild(t);
  host.appendChild(svg);
  return svg;
}

// ---------------------------------------------------------------------------------------------
// Δq. The paper's argument is that accuracy cannot tell you whether a model read the question, and
// the only honest way to show that is to let the reader try to tell. So the figure opens flat --
// every model on one line, ranked by accuracy, which is the view you get from a leaderboard -- and
// the control drops in on a button. Six variants that looked like a gradient fall onto zero.
export function mountDeltaQ(host, doc) {
  if (!host || !doc?.points?.length) return;
  const hue = { blind: OTHER, readout: OURS, teacher: MID };
  const draw = () => {
    const w = fitWidth(host, W);
    const M = { l: 62, r: 150, t: 26, b: 58 };
    const h = 420;
    const iw = w - M.l - M.r;
    const ih = h - M.t - M.b;
    const svg = frame(host, w, h, 'MMLU-Pro accuracy against the shuffled-question control: six candidate-blind variants sit at zero while every readout over the candidates carries .09 to .12.');
    const g = svgEl('g', { transform: `translate(${M.l},${M.t})` });
    svg.appendChild(g);

    const xs = doc.points.map((p) => p.acc);
    const x = scale([0, Math.max(...xs) * 1.12], [0, iw]);
    const y = scale([-0.05, 0.135], [ih, 0]);
    const flat = ih - 34; // the "accuracy only" line

    // the band the paper calls noise, so a reader can see what "at zero" was allowed to mean
    g.appendChild(svgEl('rect', { class: 'dq-band', x: 0, y: y(doc.band), width: iw, height: y(-doc.band) - y(doc.band), fill: STEEL, opacity: 0.16 }));
    g.appendChild(svgEl('line', { class: 'dq-band', x1: 0, y1: y(0), x2: iw, y2: y(0), stroke: MID, 'stroke-dasharray': '3 3' }));
    g.appendChild(svgText(iw + 8, y(0) + 4, 'the question made', { class: 'dq-band', 'font-size': 11, fill: MID, 'font-family': FONT_MONO }));
    g.appendChild(svgText(iw + 8, y(0) + 18, 'no difference', { class: 'dq-band', 'font-size': 11, fill: MID, 'font-family': FONT_MONO }));

    [0, 0.1, 0.2, 0.3].forEach((v) => {
      if (v > x.domain?.[1]) return;
      g.appendChild(svgEl('line', { x1: x(v), y1: ih, x2: x(v), y2: ih + 5, stroke: STEEL }));
      g.appendChild(svgText(x(v), ih + 20, num(v), { 'font-size': 11, fill: MID, 'text-anchor': 'middle', 'font-family': FONT_MONO }));
    });
    g.appendChild(svgText(iw / 2, ih + 42, doc.axes.x, { 'font-size': 11, fill: MID, 'text-anchor': 'middle', 'font-family': FONT_MONO }));
    [-0.05, 0, 0.05, 0.1].forEach((v) => {
      g.appendChild(svgText(-10, y(v) + 4, num(v), { class: 'dq-band', 'font-size': 11, fill: MID, 'text-anchor': 'end', 'font-family': FONT_MONO }));
    });
    g.appendChild(svgText(-M.l + 4, -10, doc.axes.y, { class: 'dq-band', 'font-size': 11, fill: MID, 'font-family': FONT_MONO }));

    // eleven of twelve points sit in two tight clusters, so the labels have to be placed rather
    // than offset. Same sweep the field scatter uses; it is self-tested in charts.js.
    const placed = placeLabels(doc.points.map((p) => {
      const cx = x(p.acc);
      const w = p.id.length * 6.6 + 16;
      return { a: cx + 10, b: cx + 10 + w, ideal: y(p.dq) + 4 };
    }), 13, ih - 2);

    doc.points.forEach((p, idx) => {
      const cx = x(p.acc);
      const cy = y(p.dq);
      const ly = placed[idx];
      const mark = svgEl('g', { class: 'dq-pt' });
      // the offset lives in a custom property so CSS owns both states and can transition between
      // them; an SVG transform attribute has no start value for a transition to run from
      mark.style.setProperty('--dy', `${(flat - cy).toFixed(1)}px`);
      mark.style.setProperty('--i', String(doc.points.indexOf(p)));
      mark.dataset.group = p.group;
      mark.appendChild(svgEl('circle', { cx, cy, r: p.group === 'blind' ? 6 : 5.5, fill: hue[p.group], opacity: p.group === 'teacher' ? 0.65 : 1 }));
      if (Math.abs(ly - (cy + 4)) > 2) {
        mark.appendChild(svgEl('polyline', {
          points: `${cx + 6},${cy} ${cx + 7},${cy} ${cx + 7},${ly - 4} ${cx + 10},${ly - 4}`,
          fill: 'none', stroke: hue[p.group], 'stroke-width': 1, opacity: 0.5,
        }));
      }
      mark.appendChild(svgText(cx + 10, ly, p.id, { 'font-size': 11, fill: hue[p.group], 'font-family': FONT_MONO, stroke: '#f0eeeb', 'stroke-width': 3, 'paint-order': 'stroke' }));
      const tip = svgEl('title');
      tip.textContent = `${p.id} — ${p.note}\naccuracy ${num(p.acc)} · Δq ${num(p.dq, 3)}`;
      mark.appendChild(tip);
      g.appendChild(mark);
    });
    host.classList.add('dq');
    return () => host.classList.toggle('is-open');
  };
  let toggle = draw();
  registerChart(host, () => { host.classList.remove('is-open'); toggle = draw(); });
  return { open: () => toggle() };
}

// ---------------------------------------------------------------------------------------------
// The readout comparison. A table, because it is a table -- nine measures across six arms is not a
// chart. What it gets instead is a per-row bar behind the number, so the shape of each row is
// readable without doing arithmetic across it.
export function mountReadouts(host, doc) {
  if (!host || !doc?.rows?.length) return;
  const table = document.createElement('table');
  table.className = 'register readouts';
  if (doc.note) {
    const cap = document.createElement('caption');
    cap.className = 'sr-only';
    cap.textContent = doc.note;
    table.appendChild(cap);
  }
  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  ['', ...doc.cols].forEach((c) => {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = c;
    if (/^N3/.test(c)) th.className = 'is-ours';
    hr.appendChild(th);
  });
  thead.appendChild(hr);
  const tbody = document.createElement('tbody');
  doc.rows.forEach((r) => {
    const tr = document.createElement('tr');
    const rowHead = document.createElement('th');
    rowHead.scope = 'row';
    rowHead.textContent = r.k;
    tr.appendChild(rowHead);
    const vals = r.v.filter((v) => v != null);
    const hi = Math.max(...vals);
    const lo = Math.min(...vals);
    const best = r.better === 'low' ? lo : hi;
    // bar length always means better, whichever direction the row is scored in
    const frac = (v) => (r.better === 'low' ? (v ? lo / v : 0) : v / hi);
    r.v.forEach((v, i) => {
      const td = document.createElement('td');
      td.dataset.label = doc.cols[i];
      td.className = 'readout-cell';
      if (v === best) td.classList.add('is-best');
      if (/^N3/.test(doc.cols[i])) td.classList.add('is-ours');
      if (v != null) {
        const bar = document.createElement('i');
        bar.style.width = `${Math.max(2, frac(v) * 100)}%`;
        td.appendChild(bar);
      }
      const s = document.createElement('span');
      s.textContent = num(v, 3);
      td.appendChild(s);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
  host.replaceChildren(table);
}

// ---------------------------------------------------------------------------------------------
// Scaling. Trained against frozen at each backbone size, with the intervals drawn, because the
// intervals are the result: most of these pairs overlap and the page should not let a reader miss it.
export function mountLadder(host, doc, tier = 'std') {
  if (!host || !doc?.rows?.length) return;
  const draw = () => {
    const w = fitWidth(host, W);
    const M = { l: 58, r: 132, t: 22, b: 56 };
    const h = 380;
    const iw = w - M.l - M.r;
    const ih = h - M.t - M.b;
    const svg = frame(host, w, h, 'JevBench accuracy against backbone size, trained and frozen, with 95% intervals.');
    const g = svgEl('g', { transform: `translate(${M.l},${M.t})` });
    svg.appendChild(g);
    const x = logScale([1.4, 17], [0, iw]);
    const y = scale([0.3, 1.0], [ih, 0]);

    [0.4, 0.6, 0.8, 1.0].forEach((v) => {
      g.appendChild(svgEl('line', { x1: 0, y1: y(v), x2: iw, y2: y(v), stroke: STEEL, 'stroke-dasharray': '1 4' }));
      g.appendChild(svgText(-10, y(v) + 4, num(v), { 'font-size': 11, fill: MID, 'text-anchor': 'end', 'font-family': FONT_MONO }));
    });
    [1.7, 4, 8, 14].forEach((v) => {
      g.appendChild(svgText(x(v), ih + 20, `${v}B`, { 'font-size': 11, fill: MID, 'text-anchor': 'middle', 'font-family': FONT_MONO }));
    });
    g.appendChild(svgText(iw / 2, ih + 42, 'backbone parameters (log)', { 'font-size': 11, fill: MID, 'text-anchor': 'middle', 'font-family': FONT_MONO }));
    g.appendChild(svgText(-M.l + 4, -8, `JevBench ${tier === 'std' ? 'standard' : 'hard'}`, { 'font-size': 11, fill: MID, 'font-family': FONT_MONO }));

    // Four rows share the 4B column (two trained, two frozen), so a fixed two-sided offset stacked
    // them. Spread each size group evenly instead, trained left of centre and frozen right.
    const at = {};
    doc.rows.forEach((r) => {
      if (r[tier] == null) return;
      const k = `${r.params}|${r.frozen ? 'f' : 't'}`;
      (at[k] = at[k] || []).push(r);
    });
    doc.rows.forEach((r) => {
      const v = r[tier];
      const ci = r[`${tier}_ci`];
      if (v == null) return;
      const peers = at[`${r.params}|${r.frozen ? 'f' : 't'}`];
      const n = peers.indexOf(r);
      const side = r.frozen ? 1 : -1;
      const cx = x(r.params) + side * (7 + n * 11);
      const col = r.frozen ? MID : (r.ships ? OURS : INK);
      const mark = svgEl('g', { class: 'lad-pt' });
      if (ci) g.appendChild(svgEl('line', { x1: cx, y1: y(ci[0]), x2: cx, y2: y(ci[1]), stroke: col, 'stroke-width': 1, opacity: 0.45 }));
      mark.appendChild(r.frozen
        ? svgEl('circle', { cx, cy: y(v), r: 4, fill: '#f0eeeb', stroke: MID, 'stroke-width': 1.4 })
        : svgEl('circle', { cx, cy: y(v), r: r.ships ? 6 : 5, fill: col }));
      const tip = svgEl('title');
      tip.textContent = `${r.name} · ${r.label}\n${tier === 'std' ? 'standard' : 'hard'} ${num(v, 3)}${ci ? ` [${num(ci[0], 3)}, ${num(ci[1], 3)}]` : ''}`;
      mark.appendChild(tip);
      g.appendChild(mark);
      if (r.ships || r.best) {
        g.appendChild(svgText(cx, y(v) - 13, r.label, {
          'font-size': 11, fill: col, 'font-weight': 700, 'text-anchor': 'middle', 'font-family': FONT_MONO,
          stroke: '#f0eeeb', 'stroke-width': 3, 'paint-order': 'stroke',
        }));
      }
    });

    const key = [['trained', INK], ['ships', OURS], ['frozen, 3-shot', MID]];
    key.forEach(([txt, c], i) => {
      const ky = 6 + i * 18;
      g.appendChild(svgEl('circle', { cx: iw + 18, cy: ky - 4, r: 4.5, fill: c === MID ? '#f0eeeb' : c, stroke: c, 'stroke-width': 1.4 }));
      g.appendChild(svgText(iw + 30, ky, txt, { 'font-size': 11, fill: MID, 'font-family': FONT_MONO }));
    });
  };
  draw();
  registerChart(host, draw);
}

export function selfTest() {
  console.assert(num(0.108) === '.11' && num(-0.026, 3) === '−.026', 'a signed value keeps its sign and drops its leading zero');
  console.assert(num(null) === '—', 'a missing cell says so');
  console.log('paperfigs.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) selfTest();

// ---------------------------------------------------------------------------------------------
// Depth. Two arms that differ only in where the head reads, on five sets. A paired dumbbell rather
// than grouped bars: the quantity the section is about is the move, not either endpoint.
export function mountDepth(host, doc) {
  if (!host || !doc?.rows?.length) return;
  const draw = () => {
    const w = fitWidth(host, 680);
    const M = { l: 108, r: 64, t: 30, b: 34 };
    const rowH = 38;
    const h = M.t + doc.rows.length * rowH + M.b;
    const iw = w - M.l - M.r;
    const svg = frame(host, w, h, 'Reading at layer 20 instead of all 28 moves every evidence set up.');
    const g = svgEl('g', { transform: `translate(${M.l},${M.t})` });
    svg.appendChild(g);
    const lo = Math.min(...doc.rows.flatMap((r) => r.v)) - 0.06;
    const x = scale([lo, 1], [0, iw]);

    doc.arms.forEach((a, i) => {
      g.appendChild(svgEl('circle', { cx: i ? 130 : 0, cy: -14, r: i ? 5 : 4, fill: i ? OURS : '#f0eeeb', stroke: i ? OURS : MID, 'stroke-width': 1.4 }));
      g.appendChild(svgText((i ? 130 : 0) + 11, -10, a, { 'font-size': 11, fill: MID, 'font-family': FONT_MONO }));
    });

    doc.rows.forEach((r, i) => {
      const y = i * rowH + 12;
      const [a, b] = r.v;
      g.appendChild(svgText(-12, y + 4, r.k, { 'font-size': 12, fill: INK, 'text-anchor': 'end', 'font-family': FONT_MONO }));
      g.appendChild(svgEl('line', { x1: x(a), y1: y, x2: x(b), y2: y, stroke: OURS, 'stroke-width': 1.5, opacity: 0.35 }));
      g.appendChild(svgEl('circle', { cx: x(a), cy: y, r: 4, fill: '#f0eeeb', stroke: MID, 'stroke-width': 1.4 }));
      g.appendChild(svgEl('circle', { cx: x(b), cy: y, r: 5, fill: OURS }));
      g.appendChild(svgText(x(b) + 11, y + 4, `+${Math.round((b - a) * 1000) / 10}`, { 'font-size': 11, fill: OURS, 'font-family': FONT_MONO }));
    });
  };
  draw();
  registerChart(host, draw);
}

// ---------------------------------------------------------------------------------------------
// Training render x evaluation render. A 2x2 per model: the diagonal is the matched comparison and
// the off-diagonal holds the evaluation render fixed, which is what makes it a factor and not a
// preference. Cell shade is accuracy above the floor.
export function mountRender(host, doc) {
  if (!host || !doc?.models) return;
  const draw = () => {
    const w = fitWidth(host, 680);
    const models = Object.entries(doc.models);
    const cell = Math.min(96, (w - 150) / (models.length * 2 + 0.6));
    const M = { l: 96, t: 54 };
    const h = M.t + 2 * cell + 56;
    const svg = frame(host, w, h, 'Training render against evaluation render on 605 held-out long states.');
    const g = svgEl('g', { transform: `translate(${M.l},${M.t})` });
    svg.appendChild(g);

    models.forEach(([name, grid], mi) => {
      const ox = mi * (2 * cell + 58);
      g.appendChild(svgText(ox + cell, -34, name, { 'font-size': 12, fill: INK, 'font-weight': 700, 'text-anchor': 'middle', 'font-family': FONT_MONO }));
      doc.labels.forEach((l, c) => g.appendChild(svgText(ox + c * cell + cell / 2, -12, l.replace('facts ', ''), { 'font-size': 10, fill: MID, 'text-anchor': 'middle', 'font-family': FONT_MONO })));
      if (mi === 0) g.appendChild(svgText(-12, -12, doc.axes.col, { 'font-size': 10, fill: STEEL, 'text-anchor': 'end', 'font-family': FONT_MONO }));
      grid.forEach((row, ri) => {
        if (mi === 0) g.appendChild(svgText(-12, ri * cell + cell / 2 + 4, doc.labels[ri], { 'font-size': 11, fill: MID, 'text-anchor': 'end', 'font-family': FONT_MONO }));
        row.forEach((v, ci) => {
          const t = Math.max(0, (v - doc.floor) / (1 - doc.floor));
          g.appendChild(svgEl('rect', { x: ox + ci * cell, y: ri * cell, width: cell - 3, height: cell - 3, fill: OURS, opacity: 0.12 + 0.82 * t }));
          g.appendChild(svgText(ox + ci * cell + (cell - 3) / 2, ri * cell + cell / 2 + 4, num(v, 3), {
            'font-size': 12, fill: t > 0.55 ? '#f0eeeb' : INK, 'text-anchor': 'middle', 'font-family': FONT_MONO, 'font-weight': ri === ci ? 700 : 400,
          }));
        });
      });
    });
    g.appendChild(svgText(-12, 2 * cell + 26, doc.axes.row, { 'font-size': 10, fill: STEEL, 'text-anchor': 'end', 'font-family': FONT_MONO }));
  };
  draw();
  registerChart(host, draw);
}
