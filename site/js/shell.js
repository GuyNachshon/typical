// Page shell for v4 ("Atoms"): mode probe, hero mark mosaic, results table + charts, the
// frozen-backbone control, exhibit expand/collapse (mount on first expand), and the lazy-mount
// loader for js/games/render-<name>.js. Static, placed, not kinetic — no GSAP, no reveals.
// Every number that reaches the page comes from data/*.json (precompute output) or a live
// decide() call — never an invented one.
import { decide, mode, probeHealth } from './api.js';
import { inkBars } from './ink.js';
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

function buildResultsTable(container, models, frozenDoc) {
  const frozenByTrained = new Map((frozenDoc?.rows ?? []).filter((r) => r.trained).map((r) => [r.trained, r.std]));

  const wrap = document.createElement('div');
  wrap.className = 'table-scroll';
  const table = document.createElement('table');
  table.className = 'results-table';
  const thead = document.createElement('thead');
  thead.innerHTML =
    '<tr><th>model</th><th>backbone</th><th>tap</th><th>ms K=2</th><th>ms K=32</th><th>ms K=256</th>' +
    '<th>JevBench std</th><th>frozen backbone, 3-shot (std)</th><th>JevBench easy</th><th>JevBench hard</th><th>ECE std</th>' +
    '<th>CLINC-150</th><th>SNLI</th><th>MNLI</th><th>BoolQ</th><th>PagerDuty (floor .792)</th><th>HF</th></tr>';
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  models
    .filter((m) => m.released)
    .forEach((m) => {
      const tr = document.createElement('tr');
      tr.append(
        td(m.id),
        td(m.backbone),
        td(m.tap),
        td(String(m.latency.single_ms.k2)),
        td(String(m.latency.single_ms.k32)),
        td(String(m.latency.single_ms.k256)),
        tdAcc(m.jevbench.std.acc),
        tdAcc(frozenByTrained.get(m.id)),
        tdAcc(m.jevbench.easy.acc),
        tdAcc(m.jevbench.hard.acc),
        td(fmt3(m.jevbench.std.ece)),
        tdAcc(m.topic_intent.clinc),
        tdAcc(m.nlu.snli),
        tdAcc(m.nlu.mnli),
        tdAcc(m.nlu.boolq),
        tdAcc(m.external.pagerduty)
      );
      const hfTd = document.createElement('td');
      const a = document.createElement('a');
      a.href = m.hf_url;
      a.target = '_blank';
      a.rel = 'noopener';
      a.className = 'ghost-link';
      a.textContent = 'weights →';
      hfTd.appendChild(a);
      tr.appendChild(hfTd);
      tbody.appendChild(tr);
    });
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);

  const footnote = document.createElement('p');
  footnote.className = 'chart-caption';
  footnote.textContent =
    'typical-small-preview → typical-small is .750 → .694 on JevBench standard (~1 SE, n_eff = 36), traded for typed heads and calibration (held-out score NLL 2.03 → 1.01).';
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
}

// ---- hero mark mosaic ------------------------------------------------------------------

function mountMark() {
  const el = document.querySelector('[data-mark]');
  if (!el) return;
  import('./mark.js').then((m) => m.mount(el));
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

  const ctx = { decide, mode, readout, presets, inkBars };

  mountMark();
  mountResults(models, reliabilityDoc, chanceDoc, frozenDoc);
  wireExhibits(ctx);
  mountGameScreens(ctx);
}

boot();
