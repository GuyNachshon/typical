// research.html bootstrap: mounts the data-mix (donut + corpus bars), held-out/external bars,
// JevBench-per-family bars, and the timeline chart from data/research-*.json into the .chart
// mounts scripts/build_research.py wraps in white .media panels. Charts are js/charts.js as-is
// (hand-rolled SVG, no animation). No sticky TOC on this page (the Agility lane has none).
import { donut, barChart, hbarFloor, timeline, fmtPct } from './charts.js';
import { glueSeparators } from './typography.js';
import { mountExplorables } from './explorables.js';

async function loadJSON(path) {
  try {
    const res = await fetch(path);
    if (res.ok) return await res.json();
  } catch {
    // static file:// serving without a data/ dir - charts just render empty
  }
  return null;
}

function sourceNote(el, text) {
  const p = document.createElement('div');
  p.className = 'chart-source';
  p.textContent = `source: ${text}`;
  el.appendChild(p);
}

async function mountMix() {
  const donutEl = document.getElementById('chart-mix-donut');
  const barsEl = document.getElementById('chart-mix-bars');
  if (!donutEl && !barsEl) return;
  const d = await loadJSON('data/research-mix.json');
  if (!d) return;
  if (donutEl) {
    donut(donutEl, {
      segments: d.family_weights.map((f) => ({ label: f.label, value: f.weight })),
      title: 'Training-batch family weights',
    });
    sourceNote(donutEl, d.source);
  }
  if (barsEl) {
    barChart(barsEl, {
      series: [{ label: 'rows', values: d.corpus_rows.map((c) => ({ x: c.key, y: c.rows })) }],
      yLabel: 'rows',
      title: 'Corpus row counts',
      fmt: (v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : String(Math.round(v))),
    });
    sourceNote(barsEl, d.source);
  }
}

async function mountHeldout() {
  const curEl = document.getElementById('chart-heldout');
  const extEl = document.getElementById('chart-external');
  if (!curEl && !extEl) return;
  const d = await loadJSON('data/research-heldout.json');
  if (!d) return;
  if (curEl) {
    hbarFloor(curEl, { rows: d.curriculum, xLabel: 'accuracy (floor = majority baseline)', title: 'Held-out curriculum' });
    sourceNote(curEl, d.source);
  }
  if (extEl) {
    hbarFloor(extEl, { rows: d.external, xLabel: 'accuracy (floor = constant-prediction baseline)', title: 'Never-trained external sets' });
    sourceNote(extEl, d.source);
  }
}

async function mountJevFamily() {
  const el = document.getElementById('chart-jevfamily');
  if (!el) return;
  const d = await loadJSON('data/research-jevbench-family.json');
  if (!d) return;
  const fam = d['typical-small'];
  barChart(el, {
    series: [{ label: 'typical-small', values: Object.entries(fam).map(([k, v]) => ({ x: k, y: v.acc })) }],
    yLabel: 'JevBench standard accuracy',
    title: 'JevBench standard, per family (typical-small)',
    fmt: fmtPct,
  });
  sourceNote(el, d.source);
}

async function mountTimeline() {
  const el = document.getElementById('chart-timeline');
  const captionEl = document.getElementById('timeline-run-count');
  if (!el && !captionEl) return;
  const d = await loadJSON('data/research-timeline.json');
  if (!d) return;
  if (el) {
    timeline(el, { lineage: d.lineage, bugs: d.bugs, verdicts: d.verdicts, title: 'Six-day decision-log timeline' });
    sourceNote(el, 'REPORT.md §2, §3q, §6 (see per-point source in each hover title)');
  }
  if (captionEl && d.run_count != null) captionEl.textContent = String(d.run_count);
}

if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    glueSeparators(); // static prose first: the mounts below are async and must not gate it
    mountMix();
    mountHeldout();
    mountJevFamily();
    mountTimeline();
    mountExplorables();
  });
  // again once the async mounts have put their captions and legends in (glue is idempotent)
  window.addEventListener('load', () => setTimeout(() => glueSeparators(), 600));
}
