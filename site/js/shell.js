// Page shell for v4 ("Atoms"): mode probe, hero mark mosaic, results table + charts, the
// frozen-backbone control, exhibit expand/collapse (mount on first expand), and the lazy-mount
// loader for js/games/render-<name>.js. Static, placed, not kinetic — no GSAP, no reveals.
// Every number that reaches the page comes from data/*.json (precompute output) or a live
// decide() call — never an invented one.
import { decide, mode, probeHealth } from './api.js';
import { dotText, dotBars } from './dots.js';
import { lineChart, reliability, ladder } from './charts.js';

async function loadJSON(path) {
  try {
    const res = await fetch(path);
    if (res.ok) return await res.json();
  } catch {
    // static file:// or missing data/ - callers render a "no data" state
  }
  return null;
}

// readout() is a no-op sink for demo modules that still call ctx.readout(...).
function readout() {}

// ---- results: model table + charts --------------------------------------------------

function fmt3(v) {
  return v == null ? '—' : v.toFixed(3);
}

function tdAcc(value) {
  const td = document.createElement('td');
  td.className = 'acc-cell';
  td.textContent = fmt3(value);
  return td;
}

function td(text) {
  const el = document.createElement('td');
  el.textContent = text;
  return el;
}

// Split into two narrower tables (model+JevBench, model+evidence/intent) instead of one wide
// 15-column table: each fits 1200px on its own without a horizontal-scroll wrapper.
function buildResultsTable(container, models, frozenDoc) {
  const frozenByTrained = new Map((frozenDoc?.rows ?? []).filter((r) => r.trained).map((r) => [r.trained, r.std]));
  const released = models.filter((m) => m.released);

  function weightsLink(m) {
    const a = document.createElement('a');
    a.href = m.hf_url;
    a.target = '_blank';
    a.rel = 'noopener';
    a.className = 'ghost-link';
    a.textContent = 'weights →';
    const cell = document.createElement('td');
    cell.appendChild(a);
    return cell;
  }

  function buildTable(heading, headers, rowFn) {
    const col = document.createElement('div');
    col.className = 'results-table-col';
    col.appendChild(Object.assign(document.createElement('h3'), { className: 'chart-title', textContent: heading }));
    const table = document.createElement('table');
    table.className = 'results-table';
    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    headers.forEach((h) => trh.appendChild(Object.assign(document.createElement('th'), { textContent: h })));
    thead.appendChild(trh);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    released.forEach((m) => {
      const tr = document.createElement('tr');
      // data-label backs the ≤600px stacked layout (results-table td::before in style.css) —
      // each cell carries its own column header so a narrow screen can drop the table grid.
      rowFn(m).forEach((cell, i) => {
        cell.dataset.label = headers[i];
        tr.appendChild(cell);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    col.appendChild(table);
    return col;
  }

  const row = document.createElement('div');
  row.className = 'results-tables';
  row.appendChild(
    buildTable('JevBench', ['model', 'std', 'hard', 'ECE std', 'frozen 3-shot', 'ms K2→K256', 'HF'], (m) => [
      td(m.id),
      tdAcc(m.jevbench.std.acc),
      tdAcc(m.jevbench.hard.acc),
      td(fmt3(m.jevbench.std.ece)),
      tdAcc(frozenByTrained.get(m.id)),
      td(`${m.latency.single_ms.k2} → ${m.latency.single_ms.k256}`),
      weightsLink(m),
    ])
  );
  row.appendChild(
    buildTable('Evidence / intent', ['model', 'CLINC-150', 'SNLI', 'MNLI', 'BoolQ', 'PagerDuty (floor .792)'], (m) => [
      td(m.id),
      tdAcc(m.topic_intent.clinc),
      tdAcc(m.nlu.snli),
      tdAcc(m.nlu.mnli),
      tdAcc(m.nlu.boolq),
      tdAcc(m.external.pagerduty),
    ])
  );
  container.appendChild(row);

  const footnote = document.createElement('p');
  footnote.className = 'chart-caption';
  footnote.textContent =
    'JevBench easy is 1.000 for every model and is omitted. typical-small-preview → typical-small is .750 → .694 on JevBench standard (about 1 SE at n = 72, SE ≈ .058), traded for typed heads and calibration (held-out score NLL 2.03 → 1.01).';
  container.appendChild(footnote);

  if (frozenByTrained.size) {
    const frozenNote = document.createElement('p');
    frozenNote.className = 'chart-caption';
    frozenNote.textContent =
      'On the hard tier the frozen 4B (.441) beats the trained 4B (.423); training helps the standard tier at 1.7B (+.17) far more than at 4B (+.03).';
    container.appendChild(frozenNote);
  }
}

function uniqueSources(list) {
  return [...new Set(list.filter(Boolean))].join(' · ');
}

function caption(el, text) {
  if (!el) return;
  const p = document.createElement('p');
  p.className = 'chart-caption';
  p.textContent = text;
  el.appendChild(p);
}

function buildInTrainingBox(container, frozenDoc) {
  if (!container || !frozenDoc?.in_flight) return;
  container.innerHTML = '';
  const strong = document.createElement('strong');
  strong.textContent = 'In training. ';
  container.appendChild(strong);
  const body = document.createElement('span');
  body.textContent = `${frozenDoc.in_flight}. Held up by a state-rendering bug, now fixed: ${frozenDoc.long_state_bug}.`;
  container.appendChild(body);
}

async function mountResults(models, reliabilityDoc, chanceDoc, frozenDoc) {
  const tableEl = document.getElementById('results-table');
  if (tableEl && models) buildResultsTable(tableEl, models, frozenDoc);
  if (!models) return;

  const g1 = document.getElementById('chart-g1');
  if (g1) {
    ladder(g1, {
      points: models.map((m) => ({
        label: m.released ? m.id : m.id === 'typical-14b-ladder' ? '14B · 6A recipe, not released' : `${m.id} (not released)`,
        x: parseFloat(m.params),
        y: m.jevbench.std.acc,
        size: m.latency.single_ms.k2,
        hollow: !m.released,
      })),
      xLabel: 'params (B)',
      yLabel: 'JevBench standard accuracy',
      title: 'scaling ladder',
      refLines: chanceDoc ? [{ y: chanceDoc.std.acc, label: `chance ${chanceDoc.std.acc}` }] : [],
    });
    const chanceTxt = chanceDoc
      ? ` Chance baseline: ${chanceDoc.std.acc} standard / ${chanceDoc.easy.acc} easy / ${chanceDoc.hard.acc} hard, n=${chanceDoc.std.n} (${chanceDoc.source}).`
      : '';
    caption(g1.parentElement, `Source: ${uniqueSources(models.map((m) => m.sources.jevbench))} (accuracy); ${uniqueSources(models.map((m) => m.sources.latency))} (ms).${chanceTxt}`);
  }

  const g2 = document.getElementById('chart-g2');
  if (g2) {
    lineChart(g2, {
      series: models.map((m) => ({
        label: m.released ? m.id : `${m.id} (not released)`,
        values: ['k2', 'k32', 'k256'].map((k) => ({ x: Number(k.slice(1)), y: m.latency.single_ms[k] })),
      })),
      xLabel: 'K (log scale)',
      yLabel: 'ms',
      logX: true,
      title: 'single-decision latency vs K',
    });
    caption(g2.parentElement, `Source: ${uniqueSources(models.map((m) => m.sources.latency))}.`);
  }

  const g4 = document.getElementById('chart-g4');
  if (g4 && reliabilityDoc) {
    g4.innerHTML = '';
    [
      ['typical-small', 'std'],
      ['typical-medium', 'std'],
      ['typical-small', 'hard'],
      ['typical-medium', 'hard'],
    ].forEach(([id, tier]) => {
      const bins = (reliabilityDoc[id]?.[tier] ?? []).filter((b) => b.n > 0);
      const m = models.find((mm) => mm.id === id);
      const ece = m?.jevbench?.[tier]?.ece;
      const panel = document.createElement('div');
      panel.className = 'chart reliability-panel';
      const h4 = document.createElement('h4');
      h4.className = 'chart-title';
      h4.textContent = `${id} · ${tier === 'std' ? 'standard' : 'hard'} · ECE ${ece != null ? ece.toFixed(3) : '—'}`;
      const holder = document.createElement('div');
      panel.append(h4, holder);
      g4.appendChild(panel);
      reliability(holder, { bins, ece, title: `${id} · ${tier === 'std' ? 'standard' : 'hard'}` });
    });
    caption(document.getElementById('g4-caption'), `Source: ${reliabilityDoc.source}.`);
  }

  buildInTrainingBox(document.getElementById('in-training'), frozenDoc);
}

// ---- exhibits: click to expand, mount js/demos/<name>.js on first expand -------------

function renderOffline(el, name) {
  el.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'demo-offline';
  div.textContent = `exhibit offline — js/demos/${name}.js not mounted yet`;
  el.appendChild(div);
}

function wireExhibits(ctx) {
  document.querySelectorAll('[data-exhibit-toggle]').forEach((btn) => {
    const card = btn.closest('[data-exhibit-card]');
    const body = card?.querySelector('[data-exhibit-body]');
    const mountEl = card?.querySelector('[data-demo]');
    let mounted = false;
    btn.addEventListener('click', () => {
      if (!body) return;
      const opening = body.hidden;
      body.hidden = !opening;
      btn.textContent = opening ? 'Hide ↑' : 'Run →';
      if (opening && !mounted && mountEl) {
        mounted = true;
        const name = mountEl.dataset.demo;
        import(`./demos/${name}.js`)
          .then((m) => m.mount(mountEl, ctx))
          .catch(() => renderOffline(mountEl, name));
      }
    });
  });
}

// ---- lazy mount: js/games/render-<name>.js (games row) ------------------------------
// Each renderer exports mount(el, {decide, mode, ctx}). Only mount once a card is in view —
// two live games never share the model's one GPU/MPS slot.
function mountGameScreens(ctx) {
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        if (entry.target.dataset.mounted) return;
        entry.target.dataset.mounted = '1';
        io.unobserve(entry.target);
        const name = entry.target.dataset.game;
        import(`./games/render-${name}.js`)
          .then((m) => m.mount(entry.target, { decide: ctx.decide, mode: ctx.mode, ctx }))
          .catch(() => renderOffline(entry.target, `render-${name}`));
      });
    },
    { rootMargin: '200px' }
  );
  document.querySelectorAll('[data-game]').forEach((el) => io.observe(el));
  // Fallback for anchor jumps (nav "Demos", #demos links): mount anything already near the viewport.
  const sweep = () => {
    document.querySelectorAll('[data-game]').forEach((el) => {
      if (el.dataset.mounted) return;
      const r = el.getBoundingClientRect();
      if (r.bottom > -200 && r.top < innerHeight + 200) {
        el.dataset.mounted = '1';
        io.unobserve(el);
        import(`./games/render-${el.dataset.game}.js`)
          .then((m) => m.mount(el, { decide: ctx.decide, mode: ctx.mode, ctx }))
          .catch(() => renderOffline(el, `render-${el.dataset.game}`));
      }
    });
  };
  addEventListener('hashchange', () => setTimeout(sweep, 50));
  addEventListener('scroll', sweep, { passive: true });
}

// ---- hero mark mosaic ------------------------------------------------------------------

// ---- the dot system: wall (wordmark), numerals, primitive glyphs ----------------------

// inkBars-compatible adapter so the demo modules need no change: rows + nullP -> dotBars rows.
function inkBars(el, { rows = [], nullP = null } = {}) {
  const toRows = (rs, np) => [
    ...rs.map((r) => ({ label: r.label, p: r.p })),
    ...(np != null ? [{ label: '∅', p: np, isNull: true }] : []),
  ];
  const h = dotBars(el, toRows(rows, nullP));
  return { update: (rs, np) => h.update(toRows(rs, np)) };
}

let wall = null;
async function mountWall() {
  const el = document.getElementById('wall');
  if (!el) return;
  const replays = (await loadJSON('data/replays.json')) || {};
  // every probability the recorded demos produced, in file order: the wordmark is data
  const values = [];
  for (const e of Object.values(replays)) {
    for (const r of e.results || []) {
      for (const v of Object.values(r.probs || {})) values.push(v);
      values.push(r.p_null ?? 0);
    }
  }
  wall = dotText(el, 'TYPICAL', { dot: 22, gap: 8, values });
  const count = document.getElementById('wall-count');
  if (count) {
    const b = document.createElement('b');
    b.textContent = values.length.toLocaleString();
    count.textContent = ' recorded probabilities';
    count.prepend(b);
  }
  // static: walk the recorded stream; live: decide() results are pushed in by pushDecision()
  if (mode() !== 'live' && values.length) {
    let i = 0;
    setInterval(() => { wall.push(values.slice(i, i + 6)); i = (i + 6) % values.length; }, 900);
  }
}
// called by the live decide wrapper below
function pushDecision(res) {
  if (!wall || !res?.results) return;
  const vec = [];
  for (const r of res.results) { for (const v of Object.values(r.probs || {})) vec.push(v); vec.push(r.p_null ?? 0); }
  wall.push(vec);
}

function mountNumerals() {
  document.querySelectorAll('[data-dotnum]').forEach((el) => dotText(el, el.dataset.dotnum, { dot: 14, gap: 5 }));
}

// Choice / Noul / Score as dot glyphs: a 5-dot row with one lit, two dots, an ascending ladder.
function mountGlyphs() {
  const shapes = {
    choice: { cols: 5, rows: 1, values: [0.05, 0.05, 0.92, 0.05, 0.05] },
    noul: { cols: 2, rows: 1, values: [0.08, 0.92] },
    score: { cols: 5, rows: 1, values: [0.1, 0.25, 0.45, 0.7, 0.95] },
  };
  document.querySelectorAll('[data-glyph]').forEach((el) => {
    const sh = shapes[el.dataset.glyph];
    if (!sh) return;
    import('./dots.js').then((d) => d.dotField(el, sh.cols, sh.rows, { dot: 18, gap: 8, values: sh.values }));
  });
}

// ---- try-it: mounted immediately (not lazy behind a Run/expand toggle) since it's the
// first exhibit card and its own "Run →" button already gates the network/replay call -----

function mountTryit(ctx) {
  const el = document.querySelector('[data-demo="tryit"]');
  if (!el) return;
  import('./demos/tryit.js')
    .then((m) => m.mount(el, ctx))
    .catch(() => renderOffline(el, 'tryit'));
}

// ---- boot -------------------------------------------------------------------------

async function boot() {
  await probeHealth();

  const [presets, models, reliabilityDoc, chanceDoc, frozenDoc] = await Promise.all([
    loadJSON('data/presets.json'),
    loadJSON('data/models.json'),
    loadJSON('data/reliability.json'),
    loadJSON('data/chance.json'),
    loadJSON('data/frozen.json'),
  ]);

  const liveDecide = async (...args) => { const res = await decide(...args); pushDecision(res); return res; };
  const ctx = { decide: liveDecide, mode, readout, presets, inkBars };

  mountWall();
  mountNumerals();
  mountGlyphs();
  mountResults(models, reliabilityDoc, chanceDoc, frozenDoc);
  mountTryit(ctx);
  wireExhibits(ctx);
  mountGameScreens(ctx);
}

boot();
