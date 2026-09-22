// research.html bootstrap: the dot-matrix "Research" title, mounts the data-mix (donut +
// corpus bars), held-out/external bars, JevBench-per-family bars, and timeline charts from
// data/research-*.json, plus a scroll-spy on the left table of contents. No GSAP, no load
// animation (Atoms: "placed, not kinetic") — charts are js/charts.js as-is: monochrome,
// series told apart by dash/marker, never colour.
import { dotText } from './dots.js';
import { donut, barChart, hbarFloor, timeline } from './charts.js';

function mountTitle() {
  const el = document.getElementById('title-dots');
  if (el) dotText(el, 'RESEARCH', { dot: 14, gap: 5 });
}

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
    fmt: (v) => v.toFixed(2),
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

// ---- table of contents: highlight the section currently in view -------------------------

function wireToc() {
  const links = Array.from(document.querySelectorAll('.toc a'));
  if (!links.length) return;
  const targets = links
    .map((a) => document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1))))
    .filter(Boolean);
  if (!targets.length) return;
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const i = targets.indexOf(entry.target);
        if (i === -1) return;
        links.forEach((a) => a.classList.remove('active'));
        links[i].classList.add('active');
      });
    },
    { rootMargin: '-10% 0px -70% 0px' }
  );
  targets.forEach((t) => io.observe(t));
}

if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    mountTitle();
    wireToc();
    mountMix();
    mountHeldout();
    mountJevFamily();
    mountTimeline();
  });
}
