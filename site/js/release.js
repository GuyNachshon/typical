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
  const head = ['model', 'kind', 'size', 'standard', 'hard'];
  const keyOf = ['name', 'kind', 'params_active', 'std', 'hard'];
  table.innerHTML = `<thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead>`;
  const body = document.createElement('tbody');
  const rowEl = (r) => {
    const tr = document.createElement('tr');
    if (r.ours) tr.className = 'is-ours';
    const name = r.size ? `${r.name} · ${r.size}` : r.name;
    [name, r.note ? `${r.kind} · ${r.note}` : r.kind, r.size || '—', fmt(r.std), fmt(r.hard)].forEach((v, i) => {
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
  ['guessing', 'majority baseline', '—', fmt(doc.chance.std), fmt(doc.chance.hard)].forEach((v, i) => {
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
  note.textContent = `${doc.what} ${doc.caveats.join(' ')}`;
  host.appendChild(note);
}

const fmt = (v) => (v == null ? '—' : v.toFixed(3).replace(/^0/, ''));

document.addEventListener('DOMContentLoaded', mountField);

// Accuracy against the parameters each token actually pays for. The frontier is drawn because it
// is the honest reading: neither of our models is on it — a 400M classifier is above typical-small,
// and SemIf is above typical-medium on the same 4B base. What our models do lead is every prompted
// backbone we ran, which is the claim this chart is really for.
import { svgEl, svgText, registerChart, fitWidth, logScale, scale, INK, MID, STEEL, FONT_MONO } from './charts.js';

const D_W = 1000, D_H = 460, M = { t: 24, r: 150, b: 54, l: 52 };

function mountSizeScore(host, doc) {
  const pts = doc.rows.filter((r) => r.params_active && r.std != null);
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

    const xs = pts.map((p) => p.params_active);
    const x = logScale([Math.min(...xs) * 0.7, Math.max(...xs) * 1.4], [0, iw]);
    const y = scale([0.3, 1.0], [ih, 0]);

    [0.4, 0.6, 0.8, 1.0].forEach((v) => {
      g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y(v), y2: y(v), stroke: STEEL, 'stroke-dasharray': '2,3' }));
      g.appendChild(svgText(-10, y(v) + 3, v.toFixed(1).replace(/^0/, ''), { fill: MID, 'font-size': 11, 'text-anchor': 'end' }));
    });
    // chance is a floor, not a gridline
    g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y(doc.chance.std), y2: y(doc.chance.std), stroke: MID, 'stroke-dasharray': '4,3' }));
    g.appendChild(svgText(iw - 2, y(doc.chance.std) - 6, `guessing ${String(doc.chance.std).replace(/^0/, '')}`, { fill: MID, 'font-size': 11, 'text-anchor': 'end' }));
    [0.4, 1, 2, 4, 9, 26].forEach((v) => {
      if (v < xs.reduce((a, b) => Math.min(a, b)) * 0.7 || v > Math.max(...xs) * 1.4) return;
      g.appendChild(svgText(x(v), ih + 20, v < 1 ? `${v * 1000}M` : `${v}B`, { fill: MID, 'font-size': 11, 'text-anchor': 'middle' }));
    });
    g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: ih, y2: ih, stroke: STEEL }));
    g.appendChild(svgText(iw / 2, ih + 44, 'active parameters (log)', { fill: MID, 'font-size': 11, 'text-anchor': 'middle' }));

    // the frontier: cheapest model at or above every score to its left
    const sorted = [...pts].sort((a, b) => a.params_active - b.params_active);
    const front = [];
    let best = -1;
    sorted.forEach((p) => { if (p.std > best) { front.push(p); best = p.std; } });
    let d = '';
    front.forEach((p, i) => { d += i ? ` H${x(p.params_active)} V${y(p.std)}` : `M${x(p.params_active)} ${y(p.std)}`; });
    g.appendChild(svgEl('path', { d, fill: 'none', stroke: MID, 'stroke-width': 1, 'stroke-dasharray': '5,4' }));

    // Labels first, dots on top. Six entries sit in the 4B column within a few points of each
    // other, so the labels have to be pushed apart or they overprint; the leader line keeps each
    // one attached to its dot. Ours are placed last so nothing can be drawn over them.
    const placed = [];
    const put = (p) => {
      const cx = x(p.params_active), cy = y(p.std);
      const flip = cx > iw * 0.72;
      // a fixed horizontal window let a long label run into a distant one; measure the box the
      // text will actually occupy instead (11px mono is ~6.6px a character)
      const wpx = p.name.length * (p.ours ? 7.2 : 6.6) + 14;
      const box = (yy) => (flip ? { a: cx - 11 - wpx, b: cx - 11, y: yy } : { a: cx + 11, b: cx + 11 + wpx, y: yy });
      let ly = cy + 4;
      for (let i = 0; i < 40; i++) {
        const me = box(ly);
        const clash = placed.some((q) => Math.abs(q.y - ly) < 13 && me.a < q.b && q.a < me.b);
        if (!clash) break;
        ly += (i % 2 ? -1 : 1) * 13 * Math.ceil((i + 1) / 2);
      }
      placed.push(box(ly));
      const tx = flip ? cx - 11 : cx + 11;
      if (Math.abs(ly - (cy + 4)) > 6) {
        g.appendChild(svgEl('line', { x1: flip ? cx - 5 : cx + 5, y1: cy, x2: tx, y2: ly - 4, stroke: STEEL, 'stroke-width': 1 }));
      }
      g.appendChild(svgText(tx, ly, p.name, {
        fill: p.ours ? INK : MID, 'font-size': p.ours ? 12 : 11, 'font-family': FONT_MONO,
        'font-weight': p.ours ? 700 : 400, 'text-anchor': flip ? 'end' : 'start',
      }));
      g.appendChild(p.ours
        ? svgEl('circle', { cx, cy, r: 6, fill: INK })
        : svgEl('circle', { cx, cy, r: 4.5, fill: '#f0eeeb', stroke: MID, 'stroke-width': 1.4 }));
    };
    pts.filter((p) => !p.ours).sort((a, b) => b.std - a.std).forEach(put);
    pts.filter((p) => p.ours).forEach(put);
    host.appendChild(svg);
  };
  draw();
  registerChart(host, draw);
}
