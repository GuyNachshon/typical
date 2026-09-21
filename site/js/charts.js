// Hand-rolled SVG charts. No libraries. Monochrome, restyled for v3 ("Gallery"): ink
// density (fill-opacity) stands in for the old thermal colour ramp, series are told apart
// by dash pattern + marker shape rather than colour, responsive via viewBox. These charts
// render inside the dark Ink Room, so the flat marks are paper-on-ink.
const NS = 'http://www.w3.org/2000/svg';
const DIM = '#808080'; // ash
const RULE = 'rgba(255,255,255,0.14)';
const FG = '#ffffff'; // paper

const DASH = ['none', '6 3', '2 3', '9 3 2 3'];
const SHAPES = ['circle', 'square', 'triangle', 'diamond'];

function seriesStyle(i) {
  return { dash: DASH[i % DASH.length], shape: SHAPES[i % SHAPES.length] };
}

// One shape vocabulary for every series marker, so lines are readable without colour.
function markerEl(shape, cx, cy, r, fill) {
  if (shape === 'square') {
    const s = r * 1.6;
    return svgEl('rect', { x: cx - s / 2, y: cy - s / 2, width: s, height: s, fill });
  }
  if (shape === 'triangle') {
    const s = r * 1.9;
    const pts = `${cx},${cy - s * 0.62} ${cx + s * 0.56},${cy + s * 0.46} ${cx - s * 0.56},${cy + s * 0.46}`;
    return svgEl('polygon', { points: pts, fill });
  }
  if (shape === 'diamond') {
    const s = r * 1.3;
    const pts = `${cx},${cy - s} ${cx + s},${cy} ${cx},${cy + s} ${cx - s},${cy}`;
    return svgEl('polygon', { points: pts, fill });
  }
  return svgEl('circle', { cx, cy, r, fill });
}

function svgEl(tag, attrs = {}) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function svgText(x, y, str, attrs = {}) {
  const t = svgEl('text', { x, y, fill: FG, 'font-size': 11, ...attrs });
  t.textContent = str;
  return t;
}

function baseSvg(container, w, h, titleStr) {
  container.innerHTML = '';
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

// Legend swatch is a dash-pattern preview (currentColor - themed by .chart-legend CSS),
// not a colour key - matches how the series are actually told apart on the chart.
function legend(container, series) {
  if (series.length < 2) return;
  const leg = document.createElement('div');
  leg.className = 'chart-legend';
  leg.innerHTML = series
    .map((s, i) => {
      const { dash } = seriesStyle(i);
      const dashAttr = dash === 'none' ? '' : ` stroke-dasharray="${dash}"`;
      return `<span class="legend-key"><svg class="legend-swatch" width="18" height="10" viewBox="0 0 18 10"><line x1="0" y1="5" x2="18" y2="5" stroke="currentColor" stroke-width="2"${dashAttr} /></svg>${s.label}</span>`;
    })
    .join('');
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
  g.appendChild(svgEl('line', { x1: 0, x2: 0, y1: 0, y2: ih, stroke: DIM }));
  g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: ih, y2: ih, stroke: DIM }));
}

function gridRow(g, iw, y, label, fmt) {
  g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y, y2: y, stroke: RULE, 'stroke-dasharray': '2,3' }));
  g.appendChild(svgText(-8, y + 3, fmt(label), { fill: DIM, 'text-anchor': 'end' }));
}

const M = { t: 20, r: 24, b: 50, l: 56 };
const W = 640;
const H = 360;

// barChart(el, {series:[{label, values:[{x,y}]}], yLabel, fmt})
export function barChart(container, { series, yLabel = '', fmt = fmtNum, title = '' }) {
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;
  const svg = baseSvg(container, W, H, title || yLabel);
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
  cats.forEach((cat, ci) => {
    series.forEach((s, si) => {
      const v = s.values[ci]?.y ?? 0;
      const bx = ci * groupW + barW * (si + 0.5);
      const by = y(v);
      const rect = svgEl('rect', {
        x: bx,
        y: by,
        width: barW * 0.85,
        height: Math.max(0, ih - by),
        fill: FG,
        'fill-opacity': (0.3 + 0.6 * (yMax ? v / yMax : 0)).toFixed(2),
      });
      const ttl = svgEl('title');
      ttl.textContent = `${s.label} · ${cat}: ${fmt(v)}`;
      rect.appendChild(ttl);
      g.appendChild(rect);
    });
    g.appendChild(svgText(ci * groupW + groupW / 2, ih + 18, String(cat), { fill: DIM, 'text-anchor': 'middle' }));
  });

  if (yLabel) g.appendChild(svgText(-M.l + 4, -8, yLabel, { fill: DIM }));
  legend(container, series);
  return svg;
}

// lineChart(el, {series, xLabel, yLabel, logX?})
export function lineChart(container, { series, xLabel = '', yLabel = '', logX = false, fmt = fmtNum, title = '' }) {
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;
  const svg = baseSvg(container, W, H, title || yLabel);
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
  xTicks.forEach((t) => g.appendChild(svgText(x(t), ih + 18, fmt(t), { fill: DIM, 'text-anchor': 'middle' })));
  axisPair(g, iw, ih);

  series.forEach((s, si) => {
    const { dash, shape } = seriesStyle(si);
    const pts = s.values.map((v) => `${x(v.x)},${y(v.y)}`).join(' ');
    const lineAttrs = { points: pts, fill: 'none', stroke: FG, 'stroke-width': 2 };
    if (dash !== 'none') lineAttrs['stroke-dasharray'] = dash;
    g.appendChild(svgEl('polyline', lineAttrs));
    s.values.forEach((v) => {
      const dot = markerEl(shape, x(v.x), y(v.y), 3.5, FG);
      const ttl = svgEl('title');
      ttl.textContent = `${s.label}: ${fmt(v.x)}, ${fmt(v.y)}`;
      dot.appendChild(ttl);
      g.appendChild(dot);
    });
  });

  if (xLabel) g.appendChild(svgText(iw / 2, ih + 38, xLabel, { fill: DIM, 'text-anchor': 'middle' }));
  if (yLabel) g.appendChild(svgText(-M.l + 4, -8, yLabel, { fill: DIM }));
  legend(container, series);
  return svg;
}

// reliability(el, {bins:[{lo,hi,n,mean_confidence,accuracy}], ece})
export function reliability(container, { bins, ece, title = 'Reliability' }) {
  const w = 400;
  const h = 400;
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
    g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y(t), y2: y(t), stroke: RULE, 'stroke-dasharray': '2,3' }));
    g.appendChild(svgEl('line', { x1: x(t), x2: x(t), y1: 0, y2: ih, stroke: RULE, 'stroke-dasharray': '2,3' }));
    g.appendChild(svgText(-8, y(t) + 3, t.toFixed(2), { fill: DIM, 'text-anchor': 'end' }));
    g.appendChild(svgText(x(t), ih + 16, t.toFixed(2), { fill: DIM, 'text-anchor': 'middle' }));
  });
  g.appendChild(svgEl('line', { x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: DIM, 'stroke-dasharray': '4,3' }));

  const maxN = Math.max(...bins.map((b) => b.n), 1);
  const slot = iw / bins.length;
  bins.forEach((b, i) => {
    const bw = slot * 0.8 * (b.n / maxN || 0.05);
    const bx = i * slot + (slot - bw) / 2;
    const by = y(b.accuracy);
    // ink density stands in for the old confidence colour ramp - denser fill = more confident
    const rect = svgEl('rect', { x: bx, y: by, width: bw, height: Math.max(0, ih - by), fill: FG, 'fill-opacity': (0.25 + 0.65 * b.mean_confidence).toFixed(2) });
    const ttl = svgEl('title');
    ttl.textContent = `[${b.lo.toFixed(2)}–${b.hi.toFixed(2)}] n=${b.n} acc=${b.accuracy.toFixed(3)} conf=${b.mean_confidence.toFixed(3)}`;
    rect.appendChild(ttl);
    g.appendChild(rect);
    if (b.n > 0) g.appendChild(svgText(i * slot + slot / 2, by - 4, `n=${b.n}`, { fill: DIM, 'text-anchor': 'middle', 'font-size': 9 }));
  });

  axisPair(g, iw, ih);
  g.appendChild(svgText(iw / 2, ih + 36, 'confidence', { fill: DIM, 'text-anchor': 'middle' }));
  g.appendChild(svgText(-mm.l + 4, -8, 'accuracy', { fill: DIM }));
  return svg;
}

// ladder(el, {points:[{label, x: ms, y: acc, size}]})
export function ladder(container, { points, xLabel = 'latency (ms)', yLabel = 'accuracy', fmt = fmtNum, title = 'Ladder', refLines = [] }) {
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;
  const svg = baseSvg(container, W, H, title);
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const refYs = refLines.map((r) => r.y);
  const x = scale([Math.min(...xs) * 0.9, Math.max(...xs) * 1.1], [0, iw]);
  const y = scale([Math.min(...ys, ...refYs) * 0.9, Math.max(...ys) * 1.1], [ih, 0]);
  const g = svgEl('g', { transform: `translate(${M.l},${M.t})` });
  svg.appendChild(g);

  niceTicks(Math.min(...ys, ...refYs) * 0.9, Math.max(...ys) * 1.1, 6).forEach((t) => gridRow(g, iw, y(t), t, fmt));
  niceTicks(Math.min(...xs) * 0.9, Math.max(...xs) * 1.1, 5).forEach((t) =>
    g.appendChild(svgText(x(t), ih + 18, fmt(t), { fill: DIM, 'text-anchor': 'middle' }))
  );
  axisPair(g, iw, ih);

  const maxSize = Math.max(...points.map((p) => p.size ?? 1));
  points.forEach((p) => {
    const norm = (p.size ?? 1) / maxSize;
    const r = 4 + 6 * norm;
    const cx = x(p.x);
    const cy = y(p.y);
    const dot = p.hollow
      ? svgEl('circle', { cx, cy, r, fill: 'none', stroke: FG, 'stroke-width': 2, 'stroke-dasharray': '3,2' })
      : svgEl('circle', { cx, cy, r, fill: FG });
    const ttl = svgEl('title');
    ttl.textContent = `${p.label}: ${fmt(p.x)}, ${fmt(p.y)}`;
    dot.appendChild(ttl);
    g.appendChild(dot);
    // labels flip to the left of the dot near the right edge so they never clip
    const flip = cx > iw * 0.8;
    g.appendChild(svgText(flip ? cx - r - 4 : cx + r + 4, cy + 3, p.label, { fill: p.hollow ? DIM : FG, 'text-anchor': flip ? 'end' : 'start' }));
  });
  refLines.forEach((rl) => {
    g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y(rl.y), y2: y(rl.y), stroke: DIM, 'stroke-dasharray': '4,3' }));
    if (rl.label) g.appendChild(svgText(iw - 4, y(rl.y) - 5, rl.label, { fill: DIM, 'text-anchor': 'end', 'font-size': 10 }));
  });

  if (xLabel) g.appendChild(svgText(iw / 2, ih + 38, xLabel, { fill: DIM, 'text-anchor': 'middle' }));
  if (yLabel) g.appendChild(svgText(-M.l + 4, -8, yLabel, { fill: DIM }));
  return svg;
}

// donut(el, {segments:[{label, value}], title}) -- fixed-order categorical donut (never
// cycled: segment i always gets the same fill-opacity step).
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
    const path = svgEl('path', { d, fill: FG, 'fill-opacity': opacity, stroke: '#000000', 'stroke-width': 1.5 });
    const ttl = svgEl('title');
    ttl.textContent = `${s.label}: ${(frac * 100).toFixed(0)}%`;
    path.appendChild(ttl);
    g.appendChild(path);
  });
  // legend below, in fixed segment order (never cycled)
  const leg = document.createElement('div');
  leg.className = 'chart-legend donut-legend';
  leg.innerHTML = segments
    .map((s, i) => {
      const opacity = (0.3 + 0.6 * (segments.length > 1 ? i / (segments.length - 1) : 0.5)).toFixed(2);
      return `<span class="donut-key"><i style="background:${FG};opacity:${opacity}"></i>${s.label} (${((s.value / total) * 100).toFixed(0)}%)</span>`;
    })
    .join('');
  container.appendChild(leg);
  return svg;
}

// hbarFloor(el, {rows:[{label, small, medium, floor}], xLabel, title})
// horizontal grouped bars (small vs medium) with a floor tick per row.
export function hbarFloor(container, { rows, xLabel = 'accuracy', title = '' }) {
  const rowH = 34;
  const h = rows.length * rowH + 50;
  const maxLabel = Math.max(...rows.map((r) => r.label.length), 10);
  const mm = { t: 10, r: 20, b: 40, l: Math.min(320, Math.max(150, maxLabel * 6.3 + 20)) };
  const iw = 460;
  const w = iw + mm.l + mm.r;
  const svg = baseSvg(container, w, h, title);
  const g = svgEl('g', { transform: `translate(${mm.l},${mm.t})` });
  svg.appendChild(g);
  const x = scale([0, 1], [0, iw]);

  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    g.appendChild(svgEl('line', { x1: x(t), x2: x(t), y1: 0, y2: rows.length * rowH, stroke: RULE, 'stroke-dasharray': '2,3' }));
    g.appendChild(svgText(x(t), rows.length * rowH + 16, t.toFixed(2), { fill: DIM, 'text-anchor': 'middle' }));
  });

  rows.forEach((row, i) => {
    const y0 = i * rowH;
    const barH = 10;
    g.appendChild(svgText(-8, y0 + rowH / 2 + 4, row.label, { fill: FG, 'text-anchor': 'end' }));
    [
      { key: 'small', dy: 3, label: 'typical-small', opacity: 1 },
      { key: 'medium', dy: 3 + barH + 3, label: 'typical-medium', opacity: 0.5 },
    ].forEach(({ key, dy, label, opacity }) => {
      const v = row[key];
      if (v == null) return;
      const rect = svgEl('rect', { x: 0, y: y0 + dy, width: Math.max(0, x(v)), height: barH, fill: FG, 'fill-opacity': opacity });
      const ttl = svgEl('title');
      ttl.textContent = `${label} · ${row.label}: ${v.toFixed(3)}${row.floor != null ? ` (floor ${row.floor.toFixed(3)})` : ''}`;
      rect.appendChild(ttl);
      g.appendChild(rect);
    });
    if (row.floor != null) {
      const fx = x(row.floor);
      const line = svgEl('line', { x1: fx, x2: fx, y1: y0 + 1, y2: y0 + rowH - 5, stroke: FG, 'stroke-width': 2 });
      const ttl = svgEl('title');
      ttl.textContent = `floor (majority/constant-prediction) · ${row.label}: ${row.floor.toFixed(3)}`;
      line.appendChild(ttl);
      g.appendChild(line);
    }
  });

  if (xLabel) g.appendChild(svgText(iw / 2, rows.length * rowH + 34, xLabel, { fill: DIM, 'text-anchor': 'middle' }));
  const leg = document.createElement('div');
  leg.className = 'chart-legend';
  leg.textContent = 'upper bar = typical-small   ·   lower bar = typical-medium   ·   | tick = floor (majority / constant-prediction baseline)';
  container.appendChild(leg);
  return svg;
}

// timeline(el, {lineage:[{date,label,std}], bugs:[{date,label}], verdicts:[{date,label}], xDomain:[d0,d1]})
// x = date (day offset), y = JevBench standard accuracy of the lineage; bugs
// and verdicts render as ticks with hover titles above/below the line.
export function timeline(container, { lineage, bugs = [], verdicts = [], title = 'Decision-log timeline' }) {
  const w = 1040;
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
    g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y(t), y2: y(t), stroke: RULE, 'stroke-dasharray': '2,3' }));
    g.appendChild(svgText(-8, y(t) + 3, t.toFixed(2), { fill: DIM, 'text-anchor': 'end' }));
  });
  // chance baseline
  g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y(0.311), y2: y(0.311), stroke: DIM, 'stroke-dasharray': '4,3' }));
  g.appendChild(svgText(iw - 4, y(0.311) - 5, 'chance (.311)', { fill: DIM, 'text-anchor': 'end' }));

  // lineage line + numbered dots. Same-date points sit at very different x
  // but sometimes close y; rather than fight label collisions inline, each
  // dot gets a small index number and the full label lives in the legend
  // list below the chart (also satisfies "a legend is always present").
  const pts = lineage.map((p) => `${x(toDay(p.date))},${y(p.std)}`).join(' ');
  g.appendChild(svgEl('polyline', { points: pts, fill: 'none', stroke: FG, 'stroke-width': 2 }));
  lineage.forEach((p, i) => {
    const cx = x(toDay(p.date));
    const cy = y(p.std);
    const dot = svgEl('circle', { cx, cy, r: 8, fill: FG, stroke: '#000000', 'stroke-width': 1.5 });
    const ttl = svgEl('title');
    ttl.textContent = `${i + 1}. ${p.date} · ${p.label}: JevBench standard ${p.std.toFixed(3)} (${p.source})`;
    dot.appendChild(ttl);
    g.appendChild(dot);
    g.appendChild(
      svgText(cx, cy + 3.5, String(i + 1), { fill: '#000000', 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700 })
    );
  });

  // x-axis date ticks, one per day of the 6-day window
  for (let day = 0; day <= 5; day++) {
    const tx = x(day);
    g.appendChild(svgEl('line', { x1: tx, x2: tx, y1: ih, y2: ih + 5, stroke: DIM }));
    g.appendChild(svgText(tx, ih + 17, fmtDay(day), { fill: DIM, 'text-anchor': 'middle', 'font-size': 10 }));
  }

  // bug ticks below axis (square marker), verdict ticks further below (triangle marker) -
  // shape, not colour, tells the two apart.
  const bugY = ih + 36;
  bugs.forEach((b, i) => {
    const bx = x(toDay(b.date));
    g.appendChild(svgEl('line', { x1: bx, x2: bx, y1: ih, y2: bugY, stroke: FG, 'stroke-width': 1.5 }));
    const dot = markerEl('square', bx, bugY, 3, FG);
    const ttl = svgEl('title');
    ttl.textContent = `bug · ${b.date}: ${b.label} (${b.source})`;
    dot.appendChild(ttl);
    g.appendChild(dot);
  });
  g.appendChild(svgText(0, bugY + 14, 'bugs found (square ticks)', { fill: DIM, 'font-size': 10 }));

  const verdictY = bugY + 34;
  verdicts.forEach((v) => {
    const vx = x(toDay(v.date));
    g.appendChild(svgEl('line', { x1: vx, x2: vx, y1: ih, y2: verdictY, stroke: FG, 'stroke-width': 1.5, opacity: 0.7 }));
    const dot = markerEl('triangle', vx, verdictY, 3, FG);
    const ttl = svgEl('title');
    ttl.textContent = `verdict · ${v.date}: ${v.label} (${v.source})`;
    dot.appendChild(ttl);
    g.appendChild(dot);
  });
  g.appendChild(svgText(0, verdictY + 14, 'section verdicts (triangle ticks)', { fill: DIM, 'font-size': 10 }));

  axisPair(g, iw, ih);
  g.appendChild(svgText(-mm.l + 4, -10, 'JevBench standard', { fill: DIM }));

  const leg = document.createElement('ol');
  leg.className = 'timeline-legend';
  leg.innerHTML = lineage
    .map((p) => `<li><b>${p.date}</b> — ${p.label}: <b>${p.std.toFixed(3)}</b> <span class="dim">(${p.source})</span></li>`)
    .join('');
  container.appendChild(leg);

  if (bugs.length) {
    const bugLabel = document.createElement('div');
    bugLabel.className = 'timeline-legend-label';
    bugLabel.textContent = 'bugs found (red ticks above)';
    container.appendChild(bugLabel);
    const bugLeg = document.createElement('ol');
    bugLeg.className = 'timeline-legend timeline-legend-bugs';
    bugLeg.innerHTML = bugs
      .map((b) => `<li><b>${b.date}</b> — ${b.label} <span class="dim">(${b.source})</span></li>`)
      .join('');
    container.appendChild(bugLeg);
  }
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
