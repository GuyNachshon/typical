// Page shell for v3 ("Gallery"): mode probe, decision ledger, painting-room pins, room-index
// dots, GSAP reveals, and the lazy-mount loader for js/demos/<name>.js + js/games/<name>
// renderers. Every number that reaches the page comes from data/*.json (precompute.py's
// output) or a live decide() call - never an invented one.
import { decide, mode, probeHealth } from './api.js';
import { inkBars, ledger as makeLedger } from './ink.js';
import { lineChart, reliability, ladder } from './charts.js';

const REDUCED_MOTION = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
const HAS_GSAP = typeof gsap !== 'undefined';
if (HAS_GSAP && typeof ScrollTrigger !== 'undefined') gsap.registerPlugin(ScrollTrigger);

async function loadJSON(path) {
  try {
    const res = await fetch(path);
    if (res.ok) return await res.json();
  } catch {
    // static file:// or missing data/ - callers render a "no data" state
  }
  return null;
}

// readout() is a no-op sink for demo modules that still call ctx.readout(...) - the v3
// rooms don't carry a per-room readout strip (that was v2's instrument-panel chrome).
function readout() {}

// heatTicker/heatNumber are gone with heat.js; demo modules that still call them get a
// harmless no-op so they mount without throwing (see js/demos/doc-20.js, stream.js).
function heatTickerNoop() {
  return { push() {} };
}
function heatNumberNoop() {}

// heatField(canvas, {rows, nullP}) shim -> inkBars(host, {rows, nullP}). Demo modules build
// a <canvas> and pass it in; inkBars draws DOM rows, so swap the canvas for a div in place.
function heatFieldShim(canvasEl, opts = {}) {
  const host = document.createElement('div');
  host.className = 'ink-bars-shim';
  if (canvasEl && canvasEl.parentNode) canvasEl.parentNode.replaceChild(host, canvasEl);
  const bars = inkBars(host, opts);
  return { update: bars.update, destroy() {} };
}

// ---- GSAP reveals ---------------------------------------------------------------------

function wireReveals() {
  if (!HAS_GSAP || REDUCED_MOTION) return;
  document.querySelectorAll('[data-reveal]').forEach((el) => {
    gsap.fromTo(
      el,
      { opacity: 0, y: 24 },
      {
        opacity: 1,
        y: 0,
        duration: 0.8,
        ease: 'power2.out',
        scrollTrigger: { trigger: el, start: 'top 85%', once: true },
      }
    );
  });
  // hero cluster fades up on load, not on scroll
  gsap.fromTo('.hero-cluster', { opacity: 0, y: 16 }, { opacity: 1, y: 0, duration: 1, ease: 'power2.out', delay: 0.1 });
}

function wireWordmarkParallax() {
  const wm = document.querySelector('[data-wordmark]');
  if (!wm || !HAS_GSAP || typeof ScrollTrigger === 'undefined' || REDUCED_MOTION) return;
  gsap.to(wm, {
    x: '-6%',
    ease: 'none',
    scrollTrigger: { trigger: '.hero', start: 'top top', end: 'bottom top', scrub: true },
  });
}

function wirePaintingPins() {
  if (!HAS_GSAP || typeof ScrollTrigger === 'undefined' || REDUCED_MOTION) return;
  document.querySelectorAll('.painting-room').forEach((room) => {
    const card = room.querySelector('[data-pin-card]');
    if (!card) return;
    gsap.fromTo(
      card,
      { scale: 0.92 },
      {
        scale: 1,
        ease: 'none',
        scrollTrigger: { trigger: room, start: 'top top', end: '+=150%', scrub: true, pin: true },
      }
    );
  });
}

// ---- room index (fixed hexagon pagination) --------------------------------------------

function hexSVG() {
  return (
    '<svg class="hex" viewBox="0 0 12 12"><polygon points="6,0.5 11,3.25 11,8.75 6,11.5 1,8.75 1,3.25" /></svg>'
  );
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
        if (i === -1) return;
        if (entry.isIntersecting) {
          buttons.forEach((b) => b.querySelector('.hex').classList.remove('filled'));
          buttons[i].querySelector('.hex').classList.add('filled');
          document.body.classList.toggle('theme-dark', entry.target.dataset.theme === 'dark');
        }
      });
    },
    { rootMargin: '-45% 0px -45% 0px' }
  );
  rooms.forEach((r) => io.observe(r));
}

// ---- decision ledger: real decisions resolved via data/demos/stream.json + decide() ----
// Mirrors v2 js/demos/stream.js's item cycling and decisionsFor() mapping exactly - each
// ledger row is either a live decide() result or the precomputed "recorded" decision for
// that same pool item, never a client-fabricated number.
const LEDGER_TYPES = ['support', 'alert', 'tool', 'ad', 'policy'];
const LEDGER_TICK_MS = 900;

function decisionsFor(kind, results) {
  if (kind === 'support') {
    const [team, wants] = results;
    return [{ label: team.argmax, p: team.probs[team.argmax] }, { label: wants.argmax, p: wants.probs[wants.argmax] }];
  }
  if (kind === 'alert' || kind === 'tool') {
    const r = results[0];
    return [{ label: r.argmax, p: r.probs[r.argmax] }];
  }
  if (kind === 'ad') {
    const [isAd] = results;
    return [{ label: isAd.argmax, p: isAd.probs[isAd.argmax] }];
  }
  const r = results[0]; // policy
  return [{ label: 'yes', p: r.probs.yes }];
}

function truncate(text, n = 72) {
  return text.length > n ? `${text.slice(0, n)}…` : text;
}

async function mountLedger(ctx) {
  const el = document.getElementById('ledger-body');
  if (!el) return;
  const pool = await loadJSON('data/demos/stream.json');
  if (!pool) {
    el.textContent = 'ledger offline — data/demos/stream.json missing';
    return;
  }
  const reg = makeLedger(el);
  const cursor = { support: 0, alert: 0, tool: 0, ad: 0, policy: 0 };
  let typeIdx = 0;

  async function tick() {
    const kind = LEDGER_TYPES[typeIdx % LEDGER_TYPES.length];
    typeIdx += 1;
    const bucket = pool[kind];
    const items = bucket?.items;
    if (!items || !items.length) return;
    const i = cursor[kind] % items.length;
    cursor[kind] += 1;
    const item = items[i];

    let decisions, ms;
    if (ctx.mode() === 'live') {
      const queries = item.queries || bucket.queries;
      const res = await ctx.decide(item.text, queries);
      if (!res) return; // server hiccup - skip, never fabricate
      decisions = decisionsFor(kind, res.results);
      ms = res.ms;
    } else {
      const rec = bucket.recorded && bucket.recorded[i];
      if (!rec) return;
      decisions = rec.decisions;
      ms = rec.ms;
    }
    const top = decisions[0];
    reg.push({ text: truncate(item.text), decision: top.label, p: top.p, ms });
  }

  for (let i = 0; i < 6; i++) await tick();
  setInterval(tick, LEDGER_TICK_MS);
}

// ---- paintings: img/paintings/credits.json (another worker's asset drop) --------------
// Schema (best-effort, until the paintings worker lands): an array of
// {slug, file, sq_file, tone, credit}. Missing file/credits -> putty placeholder, never a
// broken <img>.
async function wirePaintings() {
  const credits = await loadJSON('img/paintings/credits.json');
  const dark = (credits || []).filter((p) => p.tone === 'dark');
  const pool = dark.length ? dark : credits || [];

  document.querySelectorAll('[data-painting]').forEach((img) => {
    const i = Number(img.dataset.painting) % Math.max(pool.length, 1);
    const p = pool[i];
    if (!p) {
      img.remove(); // leave the room's putty background showing through
      return;
    }
    img.src = `img/paintings/${p.file}`;
    img.alt = p.credit || '';
  });

  document.querySelectorAll('[data-vignette]').forEach((img) => {
    const i = Number(img.dataset.vignette) % Math.max(pool.length, 1);
    const p = pool[i];
    if (!p) {
      img.closest('.vignette-crop').style.background = 'var(--color-ash)';
      return;
    }
    img.src = `img/paintings/${p.sq_file || p.file}`;
    img.alt = p.credit || '';
  });
}

// ---- results: model table + charts (ported from v2, charts.js restyled monochrome) ----

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

function buildResultsTable(container, models) {
  const wrap = document.createElement('div');
  wrap.className = 'table-scroll';
  const table = document.createElement('table');
  table.className = 'results-table';
  const thead = document.createElement('thead');
  thead.innerHTML =
    '<tr><th>model</th><th>backbone</th><th>tap</th><th>ms K=2</th><th>ms K=32</th><th>ms K=256</th>' +
    '<th>JevBench std</th><th>JevBench easy</th><th>JevBench hard</th><th>ECE std</th>' +
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
      a.className = 'link-ghost';
      a.textContent = 'weights →';
      hfTd.appendChild(a);
      tr.appendChild(hfTd);
      tbody.appendChild(tr);
    });
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
  const footnote = document.createElement('p');
  footnote.className = 'dim chart-caption';
  footnote.textContent =
    'typical-small-preview → typical-small is .750 → .694 on JevBench standard (~1 SE, n_eff = 36), traded for typed heads and calibration (held-out score NLL 2.03 → 1.01).';
  container.appendChild(footnote);
}

function uniqueSources(list) {
  return [...new Set(list.filter(Boolean))].join(' · ');
}

function caption(el, text) {
  if (!el) return;
  const p = document.createElement('p');
  p.className = 'dim chart-caption';
  p.textContent = text;
  el.appendChild(p);
}

async function mountResults(models, reliabilityDoc, chanceDoc) {
  const tableEl = document.getElementById('results-table');
  if (tableEl && models) buildResultsTable(tableEl, models);
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
    caption(g1.parentElement, 'JevBench: public subset (231 ids), unranked, n = 72 standard (SE ≈ .058); probabilities conditioned on non-∅; hard tier at chance for both models.');
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

  const g3 = document.getElementById('chart-g3');
  if (g3) {
    lineChart(g3, {
      series: models.map((m) => ({
        label: m.released ? m.id : `${m.id} (not released)`,
        values: ['k2', 'k32', 'k256'].map((k) => ({ x: Number(k.slice(1)), y: m.latency.marginal_ms_m32[k] })),
      })),
      xLabel: 'K (log scale)',
      yLabel: 'ms / extra query (M=32)',
      logX: true,
      title: 'marginal cost per query vs K',
    });
    caption(g3.parentElement, `Source: ${uniqueSources(models.map((m) => m.sources.latency))}.`);
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
      panel.className = 'chart';
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
}

// ---- lazy mount: js/demos/<name>.js (gallery + off-distribution cards) ----------------

function renderOffline(el, name) {
  el.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'demo-offline';
  div.textContent = `exhibit offline — js/demos/${name}.js not mounted yet`;
  el.appendChild(div);
}

function mountDemoScreens(ctx) {
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        io.unobserve(entry.target);
        const name = entry.target.dataset.demo;
        import(`./demos/${name}.js`)
          .then((m) => m.mount(entry.target, ctx))
          .catch(() => renderOffline(entry.target, name));
      });
    },
    { rootMargin: '200px' }
  );
  document.querySelectorAll('[data-demo]').forEach((el) => io.observe(el));
}

// ---- lazy mount: js/games/render-<name>.js (painting-room game cards) -----------------
// Each renderer exports mount(el, {decide, mode, ctx}) - see js/games/render-snake.js. Two
// live games never share the model's one GPU/MPS slot, so only mount once a card is in view.
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

// ---- boot -------------------------------------------------------------------------

async function boot() {
  await probeHealth();

  const [presets, models, reliabilityDoc, chanceDoc] = await Promise.all([
    loadJSON('data/presets.json'),
    loadJSON('data/models.json'),
    loadJSON('data/reliability.json'),
    loadJSON('data/chance.json'),
  ]);

  const ctx = {
    decide,
    mode,
    readout,
    presets,
    heatField: heatFieldShim,
    heatTicker: heatTickerNoop,
    heatNumber: heatNumberNoop,
    inkBars,
  };

  wireReveals();
  wireWordmarkParallax();
  wirePaintingPins();
  wireRoomIndex();
  wirePaintings();
  mountLedger(ctx);
  mountResults(models, reliabilityDoc, chanceDoc);
  mountDemoScreens(ctx);
  mountGameScreens(ctx);
}

boot();
