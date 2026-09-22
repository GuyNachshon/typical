// Hand-rolled SVG charts. No libraries. v11 neutral palette (designs/AGILITY_DESIGN.md) on a
// white chart surface: off-black ink for primary marks, mid-gray for axis text and a secondary
// series, steel for gridlines/rules. Series are told apart by dash pattern + marker shape (never
// colour alone, colour is only ever off-black vs mid-gray) — see seriesStyleFor below, keyed off
// the model name in the label: typical-small solid, typical-medium dashed+square, preview
// dotted+mid-gray, an unreleased model (14B) a hollow dashed outline.
const NS = 'http://www.w3.org/2000/svg';
const INK = '#292827'; // off-black — primary marks, primary text
const MID = '#938f89'; // mid-gray — axis/legend text, secondary series
const STEEL = '#c2bfba'; // steel — gridlines, axis rules, reference lines... except ref lines are mid-gray per spec
const WHITE = '#ffffff';
const FONT_MONO = '"Geist Mono", ui-monospace, "SF Mono", Menlo, monospace';

// Fixed style slots (never cycled by data order) plus a name-keyed lookup so the same model
// always gets the same treatment across every chart on the page.
const STYLE_SMALL = { color: INK, dash: 'none', shape: 'circle', hollow: false };
const STYLE_MEDIUM = { color: INK, dash: '6 3', shape: 'square', hollow: false };
const STYLE_PREVIEW = { color: MID, dash: '2 3', shape: 'circle', hollow: false };
const STYLE_UNRELEASED = { color: INK, dash: '1 5', shape: 'circle', hollow: true };
const STYLE_SLOTS = [STYLE_SMALL, STYLE_MEDIUM, STYLE_PREVIEW, STYLE_UNRELEASED];

function seriesStyleFor(label, i = 0) {
  const l = String(label || '').toLowerCase();
  if (l.includes('preview')) return STYLE_PREVIEW;
  if (l.includes('14b') || l.includes('not released')) return STYLE_UNRELEASED;
  if (l.includes('medium')) return STYLE_MEDIUM;
  if (l.includes('small')) return STYLE_SMALL;
  return STYLE_SLOTS[i % STYLE_SLOTS.length]; // generic (non-model) series: fixed order, never re-cycled per filter
}

// One shape vocabulary for every series marker, so lines are readable without colour. `hollow`
// draws an outline only (14B / not-released), matching the model's fill state everywhere else.
function markerEl(shape, cx, cy, r, fill, hollow = false) {
  const style = hollow ? { fill: 'none', stroke: fill, 'stroke-width': 1.6 } : { fill };
  if (shape === 'square') {
    const s = r * 1.6;
    return svgEl('rect', { x: cx - s / 2, y: cy - s / 2, width: s, height: s, ...style });
  }
  return svgEl('circle', { cx, cy, r, ...style });
}

function svgEl(tag, attrs = {}) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function svgText(x, y, str, attrs = {}) {
  const t = svgEl('text', { x, y, fill: INK, 'font-size': 12, 'font-family': FONT_MONO, ...attrs });
  t.textContent = str;
  return t;
}

// Charts used to draw at a fixed-width viewBox and let CSS (`width:100%`) scale the whole
// coordinate system — including text, declared in absolute px — down to fit the container. At a
// 390px container that shrank 12px labels to 2.8-5.4px. Fix: draw at the container's real width
// (fitWidth) so the viewBox always matches the rendered size 1:1 and text renders at its
// declared size. One shared, debounced ResizeObserver redraws every mounted chart on width
// change instead of one observer per chart.
const chartRedraws = new Map(); // container -> redraw fn
const lastWidths = new WeakMap(); // container -> last-drawn clientWidth
let resizeTimer = null;
let sharedRO = null;

function scheduleRedraw() {
  if (resizeTimer) clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    resizeTimer = null;
    for (const [el, redraw] of chartRedraws) {
      if (!el.isConnected) {
        chartRedraws.delete(el);
        continue;
      }
      const w = el.clientWidth;
      if (w === lastWidths.get(el)) continue; // skip when the width is unchanged
      lastWidths.set(el, w);
      redraw();
    }
  }, 120);
}

// Every chart calls this once it has drawn, so a later container resize (sidebar collapse,
// viewport change, tab reveal) redraws it at the new width.
export function registerChart(container, redraw) {
  chartRedraws.set(container, redraw);
  lastWidths.set(container, container.clientWidth);
  if (!sharedRO) sharedRO = new ResizeObserver(scheduleRedraw);
  sharedRO.observe(container);
}

// The container's real rendered width, capped at the chart's design width. No floor: a floor
// bigger than the real container would force the viewBox wider than the box CSS renders it at,
// which is exactly the bug this fixes (the box gets scaled down to fit, taking the text with it).
export function fitWidth(el, designW) {
  return Math.min(designW, el.clientWidth || designW);
}

function baseSvg(container, w, h, titleStr) {
  container.innerHTML = '';
  container.style.background = WHITE;
  // Cap to the chart's own natural width so a wide parent column (research.html's content
  // rail) can't stretch a 320px donut into an 900px blob — keeps every chart in the legible
  // 560-640px band the brief asks for (donut/reliability are intentionally narrower).
  container.style.maxWidth = `${w}px`;
  const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', role: 'img' });
  if (titleStr) {
    const t = svgEl('title');
    t.textContent = titleStr;
    svg.appendChild(t);
  }
  container.appendChild(svg);
  return svg;
}

// Default tick formatter: at most 2 decimals, no float noise.
export const fmtNum = (v) => (Number.isInteger(v) ? String(v) : Number(v.toFixed(2)).toString());
// Accuracy axes are labelled in percent (see shell.js pct): 80% reads, .804 has to be decoded.
export const fmtPct = (v) => `${Number((v * 100).toFixed(1))}%`;

// Legend key: a 10px square swatch (solid fill, or a hollow ring for an unreleased model) + the
// series label in 11px mono mid-gray — plain inline styles, this chart doesn't depend on any
// stylesheet shipping a .chart-legend rule.
function legend(container, series) {
  if (series.length < 2) return;
  const leg = document.createElement('div');
  leg.className = 'chart-legend';
  leg.style.cssText = `display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:10px;font-family:${FONT_MONO};font-size:11px;color:${MID};`;
  series.forEach((s, i) => {
    const st = seriesStyleFor(s.label, i);
    const key = document.createElement('span');
    key.className = 'legend-key';
    key.style.cssText = 'display:inline-flex;align-items:center;gap:6px;';
    const swatch = document.createElement('span');
    swatch.style.cssText = st.hollow
      ? `width:10px;height:10px;display:inline-block;border-radius:2px;border:1.6px solid ${st.color};box-sizing:border-box;`
      : `width:10px;height:10px;display:inline-block;border-radius:2px;background:${st.color};`;
    key.append(swatch, document.createTextNode(s.label));
    leg.appendChild(key);
  });
  container.appendChild(leg);
}

// scale: linear domain -> range
export function scale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
}

export function logScale(domain, range) {
  const [d0, d1] = domain.map((v) => Math.log(Math.max(v, 1e-9)));
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (v) => r0 + ((Math.log(Math.max(v, 1e-9)) - d0) / span) * (r1 - r0);
}

export function ticks(min, max, count = 5) {
  if (min === max) return [min];
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, i) => min + step * i);
}

// "Nice" ticks: round step to 1/2/5 × 10^k so labels read 0, 50, 100 rather than 43.17.
export function niceTicks(min, max, count = 5) {
  if (min === max) return [min];
  const raw = (max - min) / (count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

function axisPair(g, iw, ih) {
  g.appendChild(svgEl('line', { x1: 0, x2: 0, y1: 0, y2: ih, stroke: STEEL }));
  g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: ih, y2: ih, stroke: STEEL }));
}

function gridRow(g, iw, y, label, fmt) {
  g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y, y2: y, stroke: STEEL, 'stroke-dasharray': '2,3' }));
  g.appendChild(svgText(-8, y + 3, fmt(label), { fill: MID, 'font-size': 12, 'text-anchor': 'end' }));
}

// A reference line (chance baseline etc.): mid-gray dashed rule with an 11px mono label.
function refLine(g, iw, y, label) {
  g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y, y2: y, stroke: MID, 'stroke-dasharray': '4,3' }));
  if (label) g.appendChild(svgText(iw - 4, y - 5, label, { fill: MID, 'font-size': 11, 'text-anchor': 'end' }));
}

// A solid bar, rounded on every corner at 4px (SVG auto-clamps rx for short bars, so this stays
// correct even when a bar is thinner than 8px) — the "no dot columns" mark from here on.
function bar(g, x, y, w, h, fill, opacity = 1) {
  return svgEl('rect', { x, y, width: Math.max(0, w), height: Math.max(0, h), rx: 4, fill, 'fill-opacity': opacity });
}

const M = { t: 20, r: 24, b: 50, l: 56 };
const W_DESIGN = 640;
const H = 360;

// barChart(el, {series:[{label, values:[{x,y}]}], yLabel, fmt})
export function barChart(container, opts) {
  const { series, yLabel = '', fmt = fmtNum, title = '' } = opts;
  const w = fitWidth(container, W_DESIGN);
  const iw = w - M.l - M.r;
  const ih = H - M.t - M.b;
  const svg = baseSvg(container, w, H, title || yLabel);
  const cats = series[0]?.values.map((v) => v.x) ?? [];
  const allY = series.flatMap((s) => s.values.map((v) => v.y));
  const yMax = Math.max(...allY, 0) * 1.1 || 1;
  const y = scale([0, yMax], [ih, 0]);
  const g = svgEl('g', { transform: `translate(${M.l},${M.t})` });
  svg.appendChild(g);

  niceTicks(0, yMax, 5).forEach((t) => gridRow(g, iw, y(t), t, fmt));
  axisPair(g, iw, ih);

  const groupW = iw / Math.max(cats.length, 1);
  const barW = groupW / (series.length + 1);
  // Thin category labels when the group is narrower than the longest label needs (11px mono
  // ≈ 6.5px/char) — at 390px this stops 6-8 category names from overlapping into mud.
  const maxLabelLen = Math.max(...cats.map((c) => String(c).length), 1);
  const labelStep = Math.max(1, Math.ceil((maxLabelLen * 6.5 + 6) / groupW));
  cats.forEach((cat, ci) => {
    series.forEach((s, si) => {
      const v = s.values[ci]?.y ?? 0;
      const bw = barW * 0.85;
      const bx = ci * groupW + barW * (si + 0.5);
      const by = y(v);
      const fill = seriesStyleFor(s.label, si).color;
      const rect = bar(g, bx, by, bw, ih - by, fill);
      const ttl = svgEl('title');
      ttl.textContent = `${s.label} · ${cat}: ${fmt(v)}`;
      rect.appendChild(ttl);
      g.appendChild(rect);
    });
    if (ci % labelStep === 0) {
      g.appendChild(svgText(ci * groupW + groupW / 2, ih + 18, String(cat), { fill: MID, 'font-size': 12, 'text-anchor': 'middle' }));
    }
  });

  if (yLabel) g.appendChild(svgText(-M.l + 4, -8, yLabel, { fill: MID, 'font-size': 12 }));
  legend(container, series);
  registerChart(container, () => barChart(container, opts));
  return svg;
}

// lineChart(el, {series, xLabel, yLabel, logX?})
export function lineChart(container, opts) {
  const { series, xLabel = '', yLabel = '', logX = false, fmt = fmtNum, title = '' } = opts;
  const w = fitWidth(container, W_DESIGN);
  const iw = w - M.l - M.r;
  const ih = H - M.t - M.b;
  const svg = baseSvg(container, w, H, title || yLabel);
  const allX = series.flatMap((s) => s.values.map((v) => v.x));
  const allY = series.flatMap((s) => s.values.map((v) => v.y));
  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const yMax = Math.max(...allY, 0) * 1.1 || 1;
  const x = logX ? logScale([xMin, xMax], [0, iw]) : scale([xMin, xMax], [0, iw]);
  const y = scale([0, yMax], [ih, 0]);
  const g = svgEl('g', { transform: `translate(${M.l},${M.t})` });
  svg.appendChild(g);

  niceTicks(0, yMax, 5).forEach((t) => gridRow(g, iw, y(t), t, fmt));
  // log-x: label the actual data points (K = 2, 32, 256), not interpolated nonsense
  const xTicks = logX ? [...new Set(allX)].sort((a, b) => a - b) : niceTicks(xMin, xMax, 5);
  xTicks.forEach((t) => g.appendChild(svgText(x(t), ih + 18, fmt(t), { fill: MID, 'font-size': 12, 'text-anchor': 'middle' })));
  axisPair(g, iw, ih);

  series.forEach((s, si) => {
    const { dash, shape, color, hollow } = seriesStyleFor(s.label, si);
    const pts = s.values.map((v) => `${x(v.x)},${y(v.y)}`).join(' ');
    const lineAttrs = { points: pts, fill: 'none', stroke: color, 'stroke-width': 2 };
    if (dash !== 'none') lineAttrs['stroke-dasharray'] = dash;
    g.appendChild(svgEl('polyline', lineAttrs));
    s.values.forEach((v) => {
      const dot = markerEl(shape, x(v.x), y(v.y), 4, color, hollow); // 8px markers
      const ttl = svgEl('title');
      ttl.textContent = `${s.label}: ${fmt(v.x)}, ${fmt(v.y)}`;
      dot.appendChild(ttl);
      g.appendChild(dot);
    });
  });

  if (xLabel) g.appendChild(svgText(iw / 2, ih + 38, xLabel, { fill: MID, 'font-size': 12, 'text-anchor': 'middle' }));
  if (yLabel) g.appendChild(svgText(-M.l + 4, -8, yLabel, { fill: MID, 'font-size': 12 }));
  legend(container, series);
  registerChart(container, () => lineChart(container, opts));
  return svg;
}

// reliability(el, {bins:[{lo,hi,n,mean_confidence,accuracy}], ece})
export function reliability(container, opts) {
  const { bins, ece, title = 'Reliability' } = opts;
  const w = fitWidth(container, 400);
  const h = w; // square: axes are both 0..1
  const mm = { t: 20, r: 20, b: 50, l: 50 };
  const iw = w - mm.l - mm.r;
  const ih = h - mm.t - mm.b;
  const eceLabel = ece != null ? ` (ECE ${(ece * 100).toFixed(1)}%)` : '';
  const svg = baseSvg(container, w, h, title + eceLabel);
  const x = scale([0, 1], [0, iw]);
  const y = scale([0, 1], [ih, 0]);
  const g = svgEl('g', { transform: `translate(${mm.l},${mm.t})` });
  svg.appendChild(g);

  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y(t), y2: y(t), stroke: STEEL, 'stroke-dasharray': '2,3' }));
    g.appendChild(svgEl('line', { x1: x(t), x2: x(t), y1: 0, y2: ih, stroke: STEEL, 'stroke-dasharray': '2,3' }));
    g.appendChild(svgText(-8, y(t) + 3, fmtPct(t), { fill: MID, 'font-size': 12, 'text-anchor': 'end' }));
    g.appendChild(svgText(x(t), ih + 16, fmtPct(t), { fill: MID, 'font-size': 12, 'text-anchor': 'middle' }));
  });
  // ideal-calibration diagonal: steel dashed
  g.appendChild(svgEl('line', { x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: STEEL, 'stroke-width': 1.5, 'stroke-dasharray': '4,3' }));

  const maxN = Math.max(...bins.map((b) => b.n), 1);
  const slot = iw / bins.length;
  bins.forEach((b, i) => {
    const bw = slot * 0.8 * (b.n / maxN || 0.05);
    const bx = i * slot + (slot - bw) / 2;
    const by = y(b.accuracy);
    const rect = bar(g, bx, by, bw, ih - by, INK, 0.85);
    const ttl = svgEl('title');
    ttl.textContent = `[${b.lo.toFixed(2)}–${b.hi.toFixed(2)}] n=${b.n} acc=${fmtPct(b.accuracy)} conf=${fmtPct(b.mean_confidence)}`;
    rect.appendChild(ttl);
    g.appendChild(rect);
    if (b.n > 0) g.appendChild(svgText(i * slot + slot / 2, by - 4, `n=${b.n}`, { fill: MID, 'text-anchor': 'middle', 'font-size': 11 }));
  });

  axisPair(g, iw, ih);
  g.appendChild(svgText(iw / 2, ih + 36, 'confidence', { fill: MID, 'font-size': 12, 'text-anchor': 'middle' }));
  g.appendChild(svgText(-mm.l + 4, -8, 'accuracy', { fill: MID, 'font-size': 12 }));
  registerChart(container, () => reliability(container, opts));
  return svg;
}

// ladder(el, {points:[{label, x: ms, y: acc, size}]})
export function ladder(container, opts) {
  const { points, xLabel = 'latency (ms)', yLabel = 'accuracy', fmt = fmtNum, title = 'Ladder', refLines = [] } = opts;
  const w = fitWidth(container, W_DESIGN);
  const iw = w - M.l - M.r;
  const ih = H - M.t - M.b;
  const svg = baseSvg(container, w, H, title);
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const refYs = refLines.map((r) => r.y);
  const x = scale([Math.min(...xs) * 0.9, Math.max(...xs) * 1.1], [0, iw]);
  const y = scale([Math.min(...ys, ...refYs) * 0.9, Math.max(...ys) * 1.1], [ih, 0]);
  const g = svgEl('g', { transform: `translate(${M.l},${M.t})` });
  svg.appendChild(g);

  niceTicks(Math.min(...ys, ...refYs) * 0.9, Math.max(...ys) * 1.1, 6).forEach((t) => gridRow(g, iw, y(t), t, fmt));
  niceTicks(Math.min(...xs) * 0.9, Math.max(...xs) * 1.1, 5).forEach((t) =>
    g.appendChild(svgText(x(t), ih + 18, fmt(t), { fill: MID, 'font-size': 12, 'text-anchor': 'middle' }))
  );
  axisPair(g, iw, ih);

  const maxSize = Math.max(...points.map((p) => p.size ?? 1));
  points.forEach((p) => {
    const norm = (p.size ?? 1) / maxSize;
    const r = 4 + 6 * norm; // 8px minimum marker diameter
    const cx = x(p.x);
    const cy = y(p.y);
    const style = seriesStyleFor(p.label);
    const hollow = p.hollow ?? style.hollow;
    const dot = markerEl('circle', cx, cy, r, style.color, hollow);
    if (hollow) dot.setAttribute('stroke-dasharray', '3,2');
    const ttl = svgEl('title');
    ttl.textContent = `${p.label}: ${fmt(p.x)}, ${fmt(p.y)}`;
    dot.appendChild(ttl);
    g.appendChild(dot);
    // labels flip to the left of the dot near the right edge so they never clip. Direct labels
    // always wear text ink (mid-gray), never the marker's own colour.
    const flip = cx > iw * 0.8;
    g.appendChild(svgText(flip ? cx - r - 4 : cx + r + 4, cy + 3, p.label, { fill: MID, 'text-anchor': flip ? 'end' : 'start' }));
  });
  refLines.forEach((rl) => refLine(g, iw, y(rl.y), rl.label));

  if (xLabel) g.appendChild(svgText(iw / 2, ih + 38, xLabel, { fill: MID, 'font-size': 12, 'text-anchor': 'middle' }));
  if (yLabel) g.appendChild(svgText(-M.l + 4, -8, yLabel, { fill: MID, 'font-size': 12 }));
  registerChart(container, () => ladder(container, opts));
  return svg;
}

// donut(el, {segments:[{label, value}], title}) -- fixed-order categorical donut (never
// cycled: segment i always gets the same fill-opacity step of off-black).
export function donut(container, { segments, title = 'Mix' }) {
  const w = 320;
  const h = 320;
  const cx = w / 2;
  const cy = h / 2 - 10;
  const rOuter = 100;
  const rInner = 56;
  const svg = baseSvg(container, w, h, title);
  const g = svgEl('g', {});
  svg.appendChild(g);
  const total = segments.reduce((a, s) => a + s.value, 0) || 1;
  let angle = -Math.PI / 2;
  segments.forEach((s, i) => {
    const frac = s.value / total;
    const a0 = angle;
    const a1 = angle + frac * Math.PI * 2;
    angle = a1;
    const opacity = (0.3 + 0.6 * (segments.length > 1 ? i / (segments.length - 1) : 0.5)).toFixed(2);
    const large = a1 - a0 > Math.PI ? 1 : 0;
    const x0 = cx + rOuter * Math.cos(a0);
    const y0 = cy + rOuter * Math.sin(a0);
    const x1 = cx + rOuter * Math.cos(a1);
    const y1 = cy + rOuter * Math.sin(a1);
    const xi0 = cx + rInner * Math.cos(a1);
    const yi0 = cy + rInner * Math.sin(a1);
    const xi1 = cx + rInner * Math.cos(a0);
    const yi1 = cy + rInner * Math.sin(a0);
    const d = `M ${x0} ${y0} A ${rOuter} ${rOuter} 0 ${large} 1 ${x1} ${y1} L ${xi0} ${yi0} A ${rInner} ${rInner} 0 ${large} 0 ${xi1} ${yi1} Z`;
    // 2px white ring separates adjacent wedges on the white chart surface
    const path = svgEl('path', { d, fill: INK, 'fill-opacity': opacity, stroke: WHITE, 'stroke-width': 2 });
    const ttl = svgEl('title');
    ttl.textContent = `${s.label}: ${(frac * 100).toFixed(0)}%`;
    path.appendChild(ttl);
    g.appendChild(path);
  });
  // legend below, in fixed segment order (never cycled)
  const leg = document.createElement('div');
  leg.className = 'chart-legend donut-legend';
  leg.style.cssText = 'display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:14px;justify-content:center;';
  segments.forEach((s, i) => {
    const opacity = (0.3 + 0.6 * (segments.length > 1 ? i / (segments.length - 1) : 0.5)).toFixed(2);
    const key = document.createElement('span');
    key.className = 'donut-key';
    key.style.cssText = `display:inline-flex;align-items:center;gap:8px;font-family:${FONT_MONO};font-size:11px;color:${MID};`;
    const swatch = document.createElement('i');
    swatch.style.cssText = `width:10px;height:10px;display:inline-block;border-radius:2px;background:${INK};opacity:${opacity};`;
    key.append(swatch, document.createTextNode(`${s.label} (${((s.value / total) * 100).toFixed(0)}%)`));
    leg.appendChild(key);
  });
  container.appendChild(leg);
  return svg;
}

// hbarFloor(el, {rows:[{label, small, medium, floor}], xLabel, title})
// horizontal grouped bars (small vs medium) with a floor tick per row. Solid off-black
// (typical-small) / mid-gray (typical-medium) bars at 4px radius — no dot meters.
export function hbarFloor(container, opts) {
  const { rows, xLabel = 'accuracy', title = '' } = opts;
  const rowH = 34;
  const h = rows.length * rowH + 50;
  const maxLabel = Math.max(...rows.map((r) => r.label.length), 10);
  const mm = { t: 10, r: 20, b: 40, l: Math.min(320, Math.max(150, maxLabel * 6.3 + 20)) };
  const barsDesign = 460;
  const w = fitWidth(container, barsDesign + mm.l + mm.r);
  const iw = w - mm.l - mm.r;
  const svg = baseSvg(container, w, h, title);
  const g = svgEl('g', { transform: `translate(${mm.l},${mm.t})` });
  svg.appendChild(g);
  const x = scale([0, 1], [0, iw]);

  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    g.appendChild(svgEl('line', { x1: x(t), x2: x(t), y1: 0, y2: rows.length * rowH, stroke: STEEL, 'stroke-dasharray': '2,3' }));
    g.appendChild(svgText(x(t), rows.length * rowH + 16, fmtPct(t), { fill: MID, 'font-size': 12, 'text-anchor': 'middle' }));
  });

  rows.forEach((row, i) => {
    const y0 = i * rowH;
    const barH = 10;
    g.appendChild(svgText(-8, y0 + rowH / 2 + 4, row.label, { fill: INK, 'text-anchor': 'end' }));
    [
      { key: 'small', dy: 3, label: 'typical-small', fill: STYLE_SMALL.color },
      { key: 'medium', dy: 3 + barH + 3, label: 'typical-medium', fill: MID },
    ].forEach(({ key, dy, label, fill }) => {
      const v = row[key];
      if (v == null) return;
      const vw = Math.max(0, x(v));
      const rect = bar(g, 0, y0 + dy, vw, barH, fill);
      const ttl = svgEl('title');
      ttl.textContent = `${label} · ${row.label}: ${fmtPct(v)}${row.floor != null ? ` (floor ${fmtPct(row.floor)})` : ''}`;
      rect.appendChild(ttl);
      g.appendChild(rect);
    });
    if (row.floor != null) {
      const fx = x(row.floor);
      const line = svgEl('line', { x1: fx, x2: fx, y1: y0 + 1, y2: y0 + rowH - 5, stroke: INK, 'stroke-width': 2 });
      const ttl = svgEl('title');
      ttl.textContent = `floor (majority/constant-prediction) · ${row.label}: ${fmtPct(row.floor)}`;
      line.appendChild(ttl);
      g.appendChild(line);
    }
  });

  if (xLabel) g.appendChild(svgText(iw / 2, rows.length * rowH + 34, xLabel, { fill: MID, 'font-size': 12, 'text-anchor': 'middle' }));
  const leg = document.createElement('div');
  leg.className = 'chart-legend';
  leg.style.cssText = `font-family:${FONT_MONO};font-size:11px;color:${MID};margin-top:8px;`;
  leg.textContent = 'upper bar = typical-small (off-black)   ·   lower bar = typical-medium (mid-gray)   ·   | tick = floor (majority / constant-prediction baseline)';
  container.appendChild(leg);
  registerChart(container, () => hbarFloor(container, opts));
  return svg;
}

// timeline(el, {lineage:[{date,label,std}], bugs:[{date,label}], verdicts:[{date,label}], xDomain:[d0,d1]})
// x = date (day offset), y = JevBench standard accuracy of the lineage; bugs
// and verdicts render as ticks with hover titles above/below the line.
export function timeline(container, opts) {
  const { lineage, bugs = [], verdicts = [], title = 'Decision-log timeline' } = opts;
  const w = fitWidth(container, 1040);
  const h = 460;
  const mm = { t: 34, r: 90, b: 130, l: 50 };
  const iw = w - mm.l - mm.r;
  const ih = h - mm.t - mm.b;
  const svg = baseSvg(container, w, h, title);
  const g = svgEl('g', { transform: `translate(${mm.l},${mm.t})` });
  svg.appendChild(g);

  const toDay = (iso) => {
    const d = new Date(iso + 'T00:00:00Z');
    const d0 = new Date('2026-09-16T00:00:00Z');
    return (d - d0) / 86400000;
  };
  // Fixed 6-day domain (2026-09-16 .. 09-21), not derived from the data, so
  // the lineage's earliest point (09-18) doesn't get pushed to the right
  // third of the chart and every day gets an axis tick below.
  const x = scale([0, 5], [0, iw]);
  const y = scale([0, 1], [ih, 0]);
  const fmtDay = (day) => {
    const d = new Date(Date.UTC(2026, 8, 16) + day * 86400000);
    return `${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`;
  };

  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y(t), y2: y(t), stroke: STEEL, 'stroke-dasharray': '2,3' }));
    g.appendChild(svgText(-8, y(t) + 3, fmtPct(t), { fill: MID, 'font-size': 12, 'text-anchor': 'end' }));
  });
  refLine(g, iw, y(0.311), 'chance (.311)');

  // lineage line + numbered dots. Same-date points sit at very different x
  // but sometimes close y; rather than fight label collisions inline, each
  // dot gets a small index number and the full label lives in the legend
  // list below the chart (also satisfies "a legend is always present").
  //
  // An unreleased point (the 14B scaling-ladder probe) doesn't get to look like a release the
  // lineage climbed to: its incoming segment and marker reuse ladder()'s own "hollow dashed
  // outline" convention for an unreleased model (STYLE_UNRELEASED, see chart-g1) instead of a
  // second ad hoc style, and it gets a direct mono label rather than just a number.
  const isUnreleased = (label) => /unreleased|not released/i.test(label);
  const unreleasedLabel = (label) => label.replace(/\s*\(unreleased\)\s*$/i, ' · not released');

  for (let i = 1; i < lineage.length; i++) {
    const p0 = lineage[i - 1];
    const p1 = lineage[i];
    const attrs = {
      x1: x(toDay(p0.date)), y1: y(p0.std),
      x2: x(toDay(p1.date)), y2: y(p1.std),
      stroke: INK, 'stroke-width': 2,
    };
    if (isUnreleased(p1.label)) attrs['stroke-dasharray'] = STYLE_UNRELEASED.dash;
    g.appendChild(svgEl('line', attrs));
  }
  lineage.forEach((p, i) => {
    const cx = x(toDay(p.date));
    const cy = y(p.std);
    const unreleased = isUnreleased(p.label);
    const dot = unreleased
      ? markerEl('circle', cx, cy, 8, INK, true)
      : svgEl('circle', { cx, cy, r: 8, fill: INK, stroke: WHITE, 'stroke-width': 2 });
    if (unreleased) dot.setAttribute('stroke-dasharray', '3,2'); // matches ladder()'s hollow-dot dash
    const ttl = svgEl('title');
    ttl.textContent = `${i + 1}. ${p.date} · ${p.label}: JevBench standard ${fmtPct(p.std)} (${p.source})`;
    dot.appendChild(ttl);
    g.appendChild(dot);
    if (unreleased) {
      const flip = cx > iw * 0.8;
      g.appendChild(
        svgText(flip ? cx - 12 : cx + 12, cy + 3.5, unreleasedLabel(p.label), {
          fill: MID,
          'font-size': 11,
          'text-anchor': flip ? 'end' : 'start',
        })
      );
    } else {
      g.appendChild(svgText(cx, cy + 3.5, String(i + 1), { fill: WHITE, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700 }));
    }
  });

  // x-axis date ticks, one per day of the 6-day window. Below ~40px/day a "MM-DD" label no
  // longer fits without crowding its neighbour, so thin to every other day (tick lines stay).
  const dayStep = iw / 6 < 40 ? 2 : 1;
  for (let day = 0; day <= 5; day++) {
    const tx = x(day);
    g.appendChild(svgEl('line', { x1: tx, x2: tx, y1: ih, y2: ih + 5, stroke: STEEL }));
    if (day % dayStep === 0) g.appendChild(svgText(tx, ih + 17, fmtDay(day), { fill: MID, 'text-anchor': 'middle', 'font-size': 12 }));
  }

  // bug ticks below axis (square marker), verdict ticks further below (triangle-free, circle
  // marker) — a distinct row label tells the two apart rather than a shape only readers of the
  // legend see.
  const bugY = ih + 36;
  bugs.forEach((b) => {
    const bx = x(toDay(b.date));
    g.appendChild(svgEl('line', { x1: bx, x2: bx, y1: ih, y2: bugY, stroke: INK, 'stroke-width': 1.5 }));
    const dot = markerEl('square', bx, bugY, 3, INK);
    const ttl = svgEl('title');
    ttl.textContent = `bug · ${b.date}: ${b.label} (${b.source})`;
    dot.appendChild(ttl);
    g.appendChild(dot);
  });
  g.appendChild(svgText(0, bugY + 14, w < 700 ? 'bugs found' : 'bugs found (square ticks)', { fill: MID, 'font-size': 11 }));

  const verdictY = bugY + 34;
  verdicts.forEach((v) => {
    const vx = x(toDay(v.date));
    g.appendChild(svgEl('line', { x1: vx, x2: vx, y1: ih, y2: verdictY, stroke: MID, 'stroke-width': 1.5 }));
    const dot = markerEl('circle', vx, verdictY, 3, MID);
    const ttl = svgEl('title');
    ttl.textContent = `verdict · ${v.date}: ${v.label} (${v.source})`;
    dot.appendChild(ttl);
    g.appendChild(dot);
  });
  g.appendChild(svgText(0, verdictY + 14, w < 700 ? 'section verdicts' : 'section verdicts (round ticks, mid-gray)', { fill: MID, 'font-size': 11 }));

  axisPair(g, iw, ih);
  g.appendChild(svgText(-mm.l + 4, -10, 'JevBench standard', { fill: MID, 'font-size': 12 }));

  const leg = document.createElement('ol');
  leg.className = 'timeline-legend';
  leg.innerHTML = lineage
    .map((p) => `<li><b>${p.date}</b> — ${p.label}: <b>${fmtPct(p.std)}</b> <span class="dim">(${p.source})</span></li>`)
    .join('');
  leg.style.cssText = `color:${INK};`;
  leg.querySelectorAll('.dim').forEach((el) => (el.style.color = MID));
  container.appendChild(leg);

  if (bugs.length) {
    const bugLabel = document.createElement('div');
    bugLabel.className = 'timeline-legend-label';
    bugLabel.textContent = 'bugs found (square ticks above)';
    bugLabel.style.color = MID;
    container.appendChild(bugLabel);
    const bugLeg = document.createElement('ol');
    bugLeg.className = 'timeline-legend timeline-legend-bugs';
    bugLeg.innerHTML = bugs.map((b) => `<li><b>${b.date}</b> — ${b.label} <span class="dim">(${b.source})</span></li>`).join('');
    bugLeg.style.cssText = `color:${INK};`;
    bugLeg.querySelectorAll('.dim').forEach((el) => (el.style.color = MID));
    container.appendChild(bugLeg);
  }
  registerChart(container, () => timeline(container, opts));
  return svg;
}

function selfTest() {
  const s = scale([0, 10], [0, 100]);
  console.assert(s(0) === 0 && s(10) === 100 && s(5) === 50, 'scale is linear');
  const ls = logScale([1, 100], [0, 2]);
  console.assert(Math.abs(ls(10) - 1) < 1e-9, 'logScale midpoint at sqrt(domain)');
  const t = ticks(0, 10, 5);
  console.assert(t.length === 5 && t[0] === 0 && t[4] === 10, 'ticks spans domain with count entries');
  console.assert(JSON.stringify(niceTicks(0, 172.7)) === '[0,50,100,150]', 'niceTicks rounds to 1/2/5 steps');
  console.assert(fmtNum(1.6500000001) === '1.65' && fmtNum(45) === '45', 'fmtNum rounds float noise');
  console.log('charts.js self-test OK (pure helpers only - DOM chart fns need a browser)');
  return true;
}

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
