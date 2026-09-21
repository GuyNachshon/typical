// research.html bootstrap: mounts the data-mix (donut + corpus bars), held-out/external
// bars, JevBench-per-family bars, and timeline charts from data/research-*.json, plus the
// page's own reveal/room-index wiring (a small, page-scoped copy of shell.js's - this page
// mounts no decide()-driven demos or games, so it doesn't pull in shell.js's api.js /
// painting / ledger machinery). Charts are js/charts.js as-is: monochrome, series told
// apart by dash/marker, never colour.
import { donut, barChart, hbarFloor, timeline } from './charts.js';

const REDUCED_MOTION = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
const HAS_GSAP = typeof gsap !== 'undefined';
if (HAS_GSAP && typeof ScrollTrigger !== 'undefined') gsap.registerPlugin(ScrollTrigger);

async function loadJSON(path) {
  try {
    const res = await fetch(path);
    if (res.ok) return await res.json();
  } catch {
    // static file:// serving without a data/ dir - charts just render empty
  }
  return null;
}

// ---- reveals + room index (ported from js/shell.js's wireReveals/wireRoomIndex - this
// page has no hero cluster or wordmark, so those two shell.js concerns are skipped) ------

function wireReveals() {
  if (!HAS_GSAP || REDUCED_MOTION) return;
  document.querySelectorAll('[data-reveal]').forEach((el) => {
    gsap.fromTo(
      el,
      { opacity: 0, y: 24 },
      { opacity: 1, y: 0, duration: 0.8, ease: 'power2.out', scrollTrigger: { trigger: el, start: 'top 85%', once: true } }
    );
  });
}

function hexSVG() {
  return '<svg class="hex" viewBox="0 0 12 12"><polygon points="6,0.5 11,3.25 11,8.75 6,11.5 1,8.75 1,3.25" /></svg>';
}

function wireRoomIndex() {
  const host = document.querySelector('[data-room-index]');
  const rooms = Array.from(document.querySelectorAll('.room[data-room]'));
  if (!host || !rooms.length) return;
  const buttons = rooms.map((room) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.title = room.dataset.roomLabel || room.dataset.room;
    btn.innerHTML = hexSVG();
    btn.addEventListener('click', () => room.scrollIntoView({ behavior: REDUCED_MOTION ? 'auto' : 'smooth' }));
    host.appendChild(btn);
    return btn;
  });

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        const i = rooms.indexOf(entry.target);
        if (i === -1 || !entry.isIntersecting) return;
        buttons.forEach((b) => b.querySelector('.hex').classList.remove('filled'));
        buttons[i].querySelector('.hex').classList.add('filled');
        document.body.classList.toggle('theme-dark', entry.target.dataset.theme === 'dark');
      });
    },
    { rootMargin: '-45% 0px -45% 0px' }
  );
  rooms.forEach((r) => io.observe(r));
}

// ---- charts -----------------------------------------------------------------------------

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

if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    wireReveals();
    wireRoomIndex();
    mountMix();
    mountHeldout();
    mountJevFamily();
    mountTimeline();
  });
}
