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

import { svgEl, svgText, registerChart, fitWidth, INK, MID, STEEL, OURS, OTHER } from './charts.js';

const DESIGN_W = 1080;

// The four primitives, drawn as the probability space each one owns. One panel per card, with the
// card's own heading and paragraph underneath it -- a single strip of four charts above a separate
// block of four paragraphs made the reader match them up by counting.
//
// Schematic: these are shapes, not measurements. The numbers that belong to each type are in the
// prose beside them.
//
//   mountTypeCards()   draws into every [data-type] chart div on the page

const TYPES = {
  choice: { bars: [0.12, 0.62, 0.18, 0.08], ticks: ['A', 'B', 'C', 'D'] },
  noul: { bars: [0.16, 0.84], ticks: ['no', 'yes'] },
  score: { bars: [0.03, 0.11, 0.26, 0.6], ticks: ['0', '1', '2', '3'], ordered: true },
  abstain: { bars: [0.58, 0.2, 0.09], tail: 0.13, ticks: ['A', 'B', 'C'], tailTick: 'none' },
};

export function panelFor(name) {
  return TYPES[name] || null;
}

function drawPanel(host, ty) {
  const w = fitWidth(host, 460);
  const plotH = 88;
  const h = plotH + 26;
  const n = ty.bars.length + (ty.tail != null ? 1 : 0);
  const gap = 8;
  const bw = (w - (n - 1) * gap) / n;
  const peak = Math.max(...ty.bars);
  const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', role: 'img' });

  ty.bars.forEach((v, j) => {
    const bh = Math.max(2, v * plotH);
    const isTop = v === peak;
    svg.appendChild(svgEl('rect', {
      x: j * (bw + gap), y: plotH - bh, width: bw, height: bh,
      fill: isTop ? OURS : STEEL, opacity: ty.ordered && !isTop ? 0.45 + 0.55 * v : 1,
    }));
    svg.appendChild(svgText(j * (bw + gap) + bw / 2, plotH + 16, ty.ticks[j], { 'font-size': 10, fill: MID, 'text-anchor': 'middle' }));
  });

  if (ty.tail != null) {
    const j = ty.bars.length;
    const bh = Math.max(2, ty.tail * plotH);
    // the abstention is not one of your options, so it does not share their baseline
    svg.appendChild(svgEl('rect', { x: j * (bw + gap), y: plotH - bh, width: bw, height: bh, fill: OTHER }));
    svg.appendChild(svgText(j * (bw + gap) + bw / 2, plotH + 16, ty.tailTick, { 'font-size': 10, fill: OTHER, 'text-anchor': 'middle' }));
    svg.appendChild(svgEl('line', { x1: j * (bw + gap) - gap / 2, y1: 0, x2: j * (bw + gap) - gap / 2, y2: plotH, stroke: STEEL, 'stroke-dasharray': '2 3' }));
  }
  svg.appendChild(svgEl('line', { x1: 0, y1: plotH + 1, x2: w, y2: plotH + 1, stroke: STEEL }));
  host.replaceChildren(svg);
}

export function mountTypeCards(root = document) {
  root.querySelectorAll('[data-type]').forEach((host) => {
    const ty = panelFor(host.dataset.type);
    if (!ty) return;
    const draw = () => drawPanel(host, ty);
    draw();
    registerChart(host, draw);
  });
}

export function selfTest() {
  console.assert(Object.keys(TYPES).length === 4, 'four primitives, four panels');
  console.assert(panelFor('abstain').tail != null && panelFor('choice').tail == null, 'only abstain draws a tail outside the options');
  console.assert(panelFor('score').ordered && Math.max(...panelFor('score').bars) === panelFor('score').bars[3], 'score peaks at its top level and is drawn as a ladder');
  console.assert(panelFor('noul').bars.length === 2 && panelFor('noul').ticks.join() === 'no,yes', 'a noul is two outcomes and says which');
  console.assert(panelFor('nope') === null, 'an unknown data-type draws nothing rather than guessing');
  const ordered = orderRows([
    { name: 'Jev', weights: 'hosted only', ours: false },
    { name: 'OpenJev', weights: 'downloadable', ours: false },
    { name: 'typical-small', weights: 'downloadable', ours: true },
  ]);
  console.assert(ordered[0].name === 'typical-small', 'ours leads its group');
  const pair = orderRows([{ name: 'typical-medium', params: '4B', weights: 'downloadable', ours: true }, { name: 'typical-small', params: '1.7B', weights: 'downloadable', ours: true }]);
  console.assert(pair[0].name === 'typical-small', 'and inside a group the smaller model comes first');
  console.assert(ordered[ordered.length - 1].name === 'Jev', 'and a system you cannot download sorts last');
  console.assert(cellText({ license: null }, 'license') === '—', 'an unstated cell says so rather than guessing');
  console.assert(cellText({ needs_your_labels: false }, 'needs_your_labels') === 'no', 'a stated false is not the same as unstated');
  console.assert(cellText({ needs_your_labels: null }, 'needs_your_labels') === '—', 'and silence is not a no');
  console.log('relfigs.js self-test OK');
  return true;
}


// ---------------------------------------------------------------------------------------------
// What it takes to run each system. Deliberately NOT a scoreboard: the accuracy table sits right
// above this one and repeating it here would turn an openness comparison into a second ranking.
// What this asks instead is the question an open-weights release is actually answering -- can you
// download it, what does it cost you to stand up, and does it work before you label anything.
//
// A blank cell means nobody states it. That is a real answer and it is printed as one, because the
// alternative is inferring a licence or a hardware floor from silence.
//
//   mountRunReq(container)   doc = site/data/run-requirements.json

const COLS = [
  { key: 'weights', head: 'weights' },
  { key: 'params', head: 'size' },
  { key: 'hardware', head: 'runs on' },
  { key: 'license', head: 'licence' },
  { key: 'needs_your_labels', head: 'needs your labels' },
];

// Downloadable first, then hosted; ours to the top of their group. The sort is the argument: the
// page is about open weights, so the axis the table is ordered on is openness, not score.
// Within a group, smallest first -- alphabetical put typical-medium above typical-small, which
// reads as a ranking nobody intended.
const sizeOf = (r) => {
  const m = /([\d.]+)\s*B/i.exec(r.params || '');
  return m ? Number(m[1]) : Infinity;
};

export function orderRows(rows) {
  const rank = (r) => (r.weights === 'downloadable' ? 0 : 1) * 2 + (r.ours ? 0 : 1);
  return [...rows].sort((a, b) => rank(a) - rank(b) || sizeOf(a) - sizeOf(b) || a.name.localeCompare(b.name));
}

export function cellText(row, key) {
  const v = row[key];
  if (v == null) return '—';
  if (key === 'needs_your_labels') return v ? 'yes' : 'no';
  return String(v);
}

export function mountRunReq(container, doc) {
  const rows = orderRows(doc?.rows ?? []);
  if (!container || !rows.length) return;
  const table = document.createElement('table');
  table.className = 'register runreq';
  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  ['model', ...COLS.map((c) => c.head)].forEach((h) => {
    const th = document.createElement('th');
    th.textContent = h;
    hr.appendChild(th);
  });
  thead.appendChild(hr);
  const tbody = document.createElement('tbody');
  rows.forEach((r) => {
    const tr = document.createElement('tr');
    if (r.ours) tr.className = 'is-ours';
    const name = document.createElement('td');
    name.dataset.label = 'model';
    name.textContent = r.name;
    if (r.source) name.title = r.source;
    tr.appendChild(name);
    COLS.forEach((c) => {
      const td = document.createElement('td');
      td.dataset.label = c.head;
      const txt = cellText(r, c.key);
      td.textContent = txt;
      if (txt === '—') td.className = 'is-unstated';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
  container.replaceChildren(table);
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) selfTest();
