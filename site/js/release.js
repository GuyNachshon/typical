// release.js — the launch post. Reuses shell.js (loaded as its own <script type="module"> tag,
// unchanged, right before this one) for everything it already does: the hero and card DOOM films,
// the drive/snake game screens, and the cost chart on #chart-g2 via mountResults -> costBar off
// site/data/models.json. This page reuses those same element ids on purpose so shell.js's boot()
// mounts them with no page-specific code here.
//
// The one chart shell.js doesn't own is the read-path diagram under "One state, many decisions" —
// mount it directly.
import { mountFlow } from './arch.js';

document.addEventListener('DOMContentLoaded', () => {
  const el = document.getElementById('chart-arch');
  if (el) mountFlow(el);
});

// The whole field on one benchmark. Every row is the same 231 public ids, so the table is
// comparable by construction — which is also why it is not flattering: we sit mid-field, and a
// classifier a quarter of typical-small's size is two rows above it. Printing the field we lose in
// is worth more than printing the one we win.
async function mountField() {
  const host = document.getElementById('jev-field');
  if (!host) return;
  const doc = await fetch('data/jevbench-field.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
  if (!doc?.rows?.length) return;

  const table = document.createElement('table');
  table.className = 'register field';
  const head = ['model', 'kind', 'size', 'standard'];
  const keyOf = ['name', 'kind', 'params_total', 'std'];
  table.innerHTML = `<thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead>`;
  const body = document.createElement('tbody');
  const rowEl = (r) => {
    const tr = document.createElement('tr');
    if (r.ours) tr.className = 'is-ours';
    tr.dataset.model = r.name;
    const name = r.size ? `${r.name} · ${r.size}` : r.name;
    [name, r.note ? `${r.kind} · ${r.note}` : r.kind, r.size || '—', fmt(r.std)].forEach((v, i) => {
      const td = document.createElement('td');
      td.textContent = v;
      td.dataset.label = head[i];
      tr.appendChild(td);
    });
    return tr;
  };
  // chance sits in the table rather than in a footnote: a row you cannot beat is a row
  const chance = document.createElement('tr');
  chance.className = 'is-chance';
  ['guessing', 'majority baseline', '—', fmt(doc.chance.std)].forEach((v, i) => {
    const td = document.createElement('td');
    td.textContent = v;
    td.dataset.label = head[i];
    chance.appendChild(td);
  });
  body.appendChild(chance);
  table.appendChild(body);
  // click a header to re-sort. Numeric columns descend first because the question a reader has is
  // "who is at the top of this one", and a row with no value sorts to the bottom either way rather
  // than pretending to be a zero.
  let sortKey = 'std', desc = true;
  const rerender = () => {
    const rows = [...doc.rows].sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      const c = typeof av === 'string' ? av.localeCompare(bv) : av - bv;
      return desc ? -c : c;
    });
    body.replaceChildren(...rows.map(rowEl), chance);
    [...table.querySelectorAll('th')].forEach((th, i) => {
      th.setAttribute('aria-sort', keyOf[i] === sortKey ? (desc ? 'descending' : 'ascending') : 'none');
    });
  };
  [...table.querySelectorAll('th')].forEach((th, i) => {
    th.tabIndex = 0;
    th.classList.add('is-sortable');
    const go = () => { const k = keyOf[i]; desc = k === sortKey ? !desc : k !== 'name'; sortKey = k; rerender(); };
    th.addEventListener('click', go);
    th.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  });
  rerender();
  host.replaceChildren(table);
  mountSizeScore(document.getElementById('size-score'), doc);


  const note = document.createElement('p');
  note.className = 'note mt-18';
  note.textContent = `${doc.what} ${doc.caveats.join(' ')} `;
  const more = document.createElement('a');
  more.href = 'research.html#scaling';
  more.textContent = 'How to read these intervals.';
  note.appendChild(more);
  host.appendChild(note);
}

const fmt = (v) => (v == null ? '—' : v.toFixed(3).replace(/^0/, ''));

document.addEventListener('DOMContentLoaded', mountField);

// Accuracy against the parameters each token actually pays for. The frontier is drawn because it
// is the honest reading: neither of our models is on it — a 400M classifier is above typical-small,
// and SemIf is above typical-medium on the same 4B base. What our models do lead is every prompted
// backbone we ran, which is the claim this chart is really for.
import { svgEl, svgText, registerChart, fitWidth, logScale, scale, placeLabels, INK, MID, STEEL, OURS, FONT_MONO } from './charts.js';

const D_W = 1000, D_H = 460, M = { t: 24, r: 150, b: 54, l: 52 };
// A direct label starts LABEL_DX right of its dot and its leader turns the corner at LABEL_DX-3,
// so both ends of the leader are a visible horizontal stub rather than a hairline. LINE_H is the
// vertical pitch two labels need to clear each other at 11-12px mono.
const LABEL_DX = 17, LINE_H = 13;


// What a row's size is, for every purpose in this figure. A mixture-of-experts entry counts at the
// size you have to host, not at the parameters one token activates -- 26B-A4B needs 26B resident
// however little of it any given token touches. One accessor so the axis, the ticks, the frontier
// and the exclusion count can never disagree with each other.
const sizeOfRow = (r) => r.params_total ?? r.params_active;

// The caption used to say "Five entries publish no size" and then a data edit gave one of those
// five a size, so the page carried a wrong number until someone counted by hand. It counts itself
// now: the word in the caption comes from the same array the chart excludes rows by.
const COUNT_WORDS = ['no', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function countWord(n) {
  return COUNT_WORDS[n] ?? String(n);
}

function writeUnplottedCount(doc) {
  const el = document.querySelector('[data-count="unplotted"]');
  if (!el) return;
  const n = (doc.rows || []).filter((r) => !sizeOfRow(r)).length;
  el.textContent = countWord(n);
}

// The table and the scatter are the same twenty rows twice. Point at one and the other says which
// row you are pointing at -- the only way, short of counting, to find `system-one-open` in a field
// of sixteen dots. Delegated from the section, because the chart redraws itself on every resize and
// listeners bound to its marks would not survive that.
function linkFieldAndScatter() {
  const section = document.getElementById('field');
  if (!section || section.dataset.linked) return;
  section.dataset.linked = '1';
  const set = (name) => {
    section.classList.toggle('is-linking', !!name);
    section.querySelectorAll('[data-model]').forEach((el) => {
      el.classList.toggle('is-lit', !!name && el.dataset.model === name);
    });
  };
  section.addEventListener('pointerover', (e) => {
    const hit = e.target.closest?.('[data-model]');
    set(hit ? hit.dataset.model : null);
  });
  section.addEventListener('pointerleave', () => set(null));
}

function mountSizeScore(host, doc) {
  writeUnplottedCount(doc);
  const pts = doc.rows.filter((r) => sizeOfRow(r) && r.std != null);
  if (!host || pts.length < 3) return;
  const draw = () => {
    const w = fitWidth(host, D_W), h = D_H;
    host.innerHTML = '';
    host.style.background = '#f0eeeb';
    const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', role: 'img' });
    const t = svgEl('title');
    t.textContent = 'JevBench standard accuracy against active parameters, same 231 public ids.';
    svg.appendChild(t);
    const iw = w - M.l - M.r, ih = h - M.t - M.b;
    const g = svgEl('g', { transform: `translate(${M.l},${M.t})` });
    svg.appendChild(g);

    const xs = pts.map(sizeOfRow);
    const x = logScale([Math.min(...xs) * 0.7, Math.max(...xs) * 1.4], [0, iw]);
    const y = scale([0.3, 1.0], [ih, 0]);

    [0.4, 0.6, 0.8, 1.0].forEach((v) => {
      g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y(v), y2: y(v), stroke: STEEL, 'stroke-dasharray': '2,3' }));
      g.appendChild(svgText(-10, y(v) + 3, v.toFixed(1).replace(/^0/, ''), { fill: MID, 'font-size': 11, 'text-anchor': 'end' }));
    });
    // chance is a floor, not a gridline
    g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y(doc.chance.std), y2: y(doc.chance.std), stroke: MID, 'stroke-dasharray': '4,3' }));
    g.appendChild(svgText(iw - 2, y(doc.chance.std) - 6, `guessing ${String(doc.chance.std).replace(/^0/, '')}`, { fill: MID, 'font-size': 11, 'text-anchor': 'end' }));
    // The ticks are the distinct active-parameter counts actually in the data. They used to be a
    // hand-typed [0.4, 1, 2, 4, 9, 26]: nothing here is 1B, 26 is the *total* of the two
    // mixture-of-experts rows whose tokens only pay for 4 (so it was labelling this axis with a
    // number this axis does not plot), and the list stopped at 9B while the right-most point is a
    // 14B — which is why the sizes read as wrong. Thinned left to right when a label would print
    // into its neighbour, which only happens in a narrow container (8B and 9B sit 22px apart at
    // the design width, 5px at 400px).
    let tickX = -Infinity;
    [...new Set(xs)].sort((a, b) => a - b).forEach((v) => {
      const label = v < 1 ? `${v * 1000}M` : `${v}B`;
      const cx = x(v);
      if (cx - tickX < label.length * 3.3 + 10) return;
      tickX = cx;
      g.appendChild(svgText(cx, ih + 20, label, { fill: MID, 'font-size': 11, 'text-anchor': 'middle' }));
    });
    g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: ih, y2: ih, stroke: STEEL }));
    g.appendChild(svgText(iw / 2, ih + 44, 'parameters (log)', { fill: MID, 'font-size': 11, 'text-anchor': 'middle' }));
    // The y axis had no title at all: a reader met a .4-1.0 scale and the only statement of what
    // it measures was in the SVG <title>, where nothing but a screen reader finds it. Same idiom
    // as the x title above, same place every other chart on the site puts it (barChart, ladder,
    // timeline: mono, mid-gray, above the top of the axis).
    g.appendChild(svgText(-M.l + 4, -8, 'JevBench standard (public subset)', { fill: MID, 'font-size': 11 }));

    // the frontier: cheapest model at or above every score to its left
    const sorted = [...pts].sort((a, b) => sizeOfRow(a) - sizeOfRow(b));
    const front = [];
    let best = -1;
    sorted.forEach((p) => { if (p.std > best) { front.push(p); best = p.std; } });
    let d = '';
    front.forEach((p, i) => { d += i ? ` H${x(sizeOfRow(p))} V${y(p.std)}` : `M${x(sizeOfRow(p))} ${y(p.std)}`; });
    g.appendChild(svgEl('path', { d, fill: 'none', stroke: MID, 'stroke-width': 1, 'stroke-dasharray': '5,4' }));

    // Eight entries sit in the 4B column, three of them within two points of each other, so the
    // labels have to be pushed apart. Two things were wrong with how that was done.
    //
    // (1) A label whose dot sat past 72% of the plot width flipped to the LEFT of that dot. With
    // 150px of right margin reserved for exactly these labels there was never a reason to: the
    // 14B label flipped into the span [617, 724], which is where the 9B dot (652) and the 8B dot
    // (630) live, so it read as the label of a point to its left. A label now flips only if it
    // would otherwise leave the SVG, which at any width this page renders never happens.
    //
    // (2) The vertical search stepped `ly` by a cumulative +13, -13, +26, -26 ..., so the offsets
    // it actually visited were 0, +13, 0, +26, 0, +39: never upwards, every other attempt a repeat
    // of the position that had already clashed, and up to 260px of travel. Worst case in this data
    // was 130px of drift at a 800px container. placeLabels() (charts.js, self-tested) replaces it
    // with one downward sweep in y order; nothing in this data now moves more than 19px.
    //
    // Each label keeps its dot's x, so the leader only has to say "up/down from here": a stub out
    // of the dot, a vertical run in the gutter just left of the text, a stub into the text.
    const compact = w < 520;
    const items = (compact ? pts.filter((p) => p.ours) : pts).map((p) => {
      const cx = x(sizeOfRow(p)), cy = y(p.std);
      // measure the box the text will actually occupy (11px mono is ~6.6px a character, our own
      // rows are set at 12px bold) rather than assuming a fixed window
      const wpx = p.name.length * (p.ours ? 7.2 : 6.6) + 14;
      const flip = cx + LABEL_DX + wpx > iw + M.r - 6;
      const a = flip ? cx - LABEL_DX - wpx : cx + LABEL_DX;
      return { p, cx, cy, flip, a, b: a + wpx, ideal: cy + 4 };
    });
    placeLabels(items, LINE_H, ih - 2).forEach((ly, i) => (items[i].ly = ly));

    // Labels first, dots on top; ours drawn last so nothing can be drawn over them.
    const draw = ({ p, cx, cy, flip, ly }) => {
      const s = flip ? -1 : 1;
      const tx = cx + s * LABEL_DX;
      const mark = svgEl('g', { class: 'jev-mark' });
      mark.dataset.model = p.name;
      g.appendChild(mark);
      // any displacement at all gets a leader: a label nudged 6px is exactly the one a reader
      // cannot tell is nudged
      if (Math.abs(ly - (cy + 4)) > 2) {
        const gutter = cx + s * (LABEL_DX - 3);
        mark.appendChild(svgEl('polyline', {
          points: `${cx + s * 7},${cy} ${gutter},${cy} ${gutter},${ly - 4} ${tx},${ly - 4}`,
          fill: 'none', stroke: MID, 'stroke-width': 1,
        }));
      }
      // knocked out of whatever rule it lands on — the frontier step and the .8 gridline ran
      // straight through `jeff (GLiFormer)` and `system-one-open` like a strike-through
      mark.appendChild(svgText(tx, ly, p.name, {
        fill: p.ours ? OURS : MID, 'font-size': p.ours ? 12 : 11, 'font-family': FONT_MONO,
        'font-weight': p.ours ? 700 : 400, 'text-anchor': flip ? 'end' : 'start',
        stroke: '#f0eeeb', 'stroke-width': 3, 'paint-order': 'stroke',
      }));
      mark.appendChild(p.ours
        ? svgEl('circle', { cx, cy, r: 6, fill: OURS })
        : svgEl('circle', { cx, cy, r: 4.5, fill: '#f0eeeb', stroke: MID, 'stroke-width': 1.4 }));
    };
    if (compact) pts.filter((p) => !p.ours).forEach((p) => {
      const mark = svgEl('g', { class: 'jev-mark' });
      mark.dataset.model = p.name;
      mark.appendChild(svgEl('circle', { cx: x(sizeOfRow(p)), cy: y(p.std), r: 4.5, fill: '#f0eeeb', stroke: MID, 'stroke-width': 1.4 }));
      g.appendChild(mark);
    });
    items.filter((it) => !it.p.ours).forEach(draw);
    items.filter((it) => it.p.ours).forEach(draw);
    host.appendChild(svg);
    linkFieldAndScatter();
  };
  draw();
  registerChart(host, draw);
}

import { mountShowdown } from './showdown.js';
import { mountParallel } from './parallel.js';
import { mountTypeCards, mountRunReq } from './relfigs.js';

// Figure numbers were typed into the captions by hand, so moving a section renumbered nothing and
// two figures both called themselves Fig. 1. They are numbered from document order instead, and a
// cross-reference names its target rather than a literal that has to be kept in step.
function numberFigures() {
  const figs = [...document.querySelectorAll('.post figure.fig')];
  figs.forEach((fig, i) => {
    const cap = fig.querySelector('figcaption');
    if (!cap || cap.dataset.numbered) return;
    cap.dataset.numbered = '1';
    const b = document.createElement('b');
    b.textContent = `Fig. ${i + 1}.`;
    cap.prepend(b, ' ');
    if (fig.closest('section')?.id) fig.dataset.fignum = String(i + 1);
  });
  document.querySelectorAll('a.figref').forEach((a) => {
    const target = document.querySelector(`${a.getAttribute('href')} figure.fig`);
    if (target?.dataset.fignum) a.textContent = `Fig. ${target.dataset.fignum}`;
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  numberFigures();
  mountParallel(document.getElementById('chart-parallel'));
  mountTypeCards();
  fetch('data/run-requirements.json')
    .then((r) => (r.ok ? r.json() : null))
    .then((doc) => doc && mountRunReq(document.getElementById('chart-runreq'), doc))
    .catch(() => {});
  const host = document.getElementById('showdown');
  if (!host) return;
  const doc = await fetch('data/showdown.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
  if (doc) mountShowdown(host, doc);
});
