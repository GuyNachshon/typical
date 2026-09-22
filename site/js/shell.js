// Page shell for v4 ("Atoms"): mode probe, hero mark mosaic, results table + charts, the
// frozen-backbone control, exhibit expand/collapse (mount on first expand), and the lazy-mount
// loader for js/games/render-<name>.js. Static, placed, not kinetic — no GSAP, no reveals.
// Every number that reaches the page comes from data/*.json (precompute output) or a live
// decide() call — never an invented one.
import { decide, mode, probeHealth } from './api.js';
import { bars, inkBars, rowsFromResult } from './bars.js';
import { glueSeparators, glued } from './typography.js';
import { describeDoom, candidatesFor, resolveIntent, keyPress, scriptedPolicy, resetNav } from './games/realdoom-logic.js';
import { QUESTION as DOOM_QUESTION } from './games/doom.js';
import { hashKey } from './api.js';
import { lineChart, reliability, ladder, fmtPct } from './charts.js';

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

// Accuracies and recalls are printed as percentages on this page: ".804" is a figure only a
// reader of the report recognises. Probabilities the model emits (the candidate bars, p(∅)) stay
// in probability units — those are the model's output, not a score — and so do ECE and NLL.
function pct(v, dp = 1) {
  return v == null ? '—' : `${(v * 100).toFixed(dp)}%`;
}

// .register already sets tabular-nums on the whole table (src.css) — no per-cell class needed.
function tdAcc(value) {
  const td = document.createElement('td');
  td.textContent = pct(value);
  return td;
}

function td(text) {
  const el = document.createElement('td');
  el.textContent = text;
  return el;
}

// Inject the two-column grid once (JS-owned chunk of the port — buildResultsTable doesn't own
// src.css, so the one rule it needs ships as a scoped <style>, same mechanism a component would
// use). Single column ≤1024px per the design brief.
let resultsGridStyled = false;
function ensureResultsGridStyle() {
  if (resultsGridStyled) return;
  resultsGridStyled = true;
  const style = document.createElement('style');
  style.textContent =
    '.results-tables{display:grid;grid-template-columns:1fr 1fr;gap:30px}' +
    '@media (max-width:1024px){.results-tables{grid-template-columns:1fr}}';
  document.head.appendChild(style);
}

// Split into two narrower tables (model+JevBench, model+evidence/intent) instead of one wide
// 15-column table: each fits 1200px on its own without a horizontal-scroll wrapper. Both render
// as .register (src.css: mono uppercase th, data-label stacking ≤640px) under a mono eyebrow.
function buildResultsTable(container, models, frozenDoc) {
  ensureResultsGridStyle();
  const frozenByTrained = new Map((frozenDoc?.rows ?? []).filter((r) => r.trained).map((r) => [r.trained, r.std]));
  const released = models.filter((m) => m.released);

  function weightsLink(m) {
    const a = document.createElement('a');
    a.href = m.hf_url;
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = 'weights →';
    const cell = document.createElement('td');
    cell.appendChild(a);
    return cell;
  }

  function buildTable(eyebrow, headers, rowFn) {
    const col = document.createElement('div');
    col.appendChild(Object.assign(document.createElement('p'), { className: 't-eyebrow muted', textContent: eyebrow, style: 'margin-bottom:12px' }));
    const table = document.createElement('table');
    table.className = 'register';
    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    headers.forEach((h) => trh.appendChild(Object.assign(document.createElement('th'), { textContent: h })));
    thead.appendChild(trh);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    released.forEach((m) => {
      const tr = document.createElement('tr');
      // data-label backs the ≤640px stacked layout (.register td::before in src.css) — each
      // cell carries its own column header so a narrow screen can drop the table grid.
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
    buildTable('Evidence / intent', ['model', 'CLINC-150', 'SNLI', 'MNLI', 'BoolQ', 'PagerDuty (floor 79.2%)'], (m) => [
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
  footnote.className = 'note';
  footnote.style.marginTop = '18px';
  footnote.textContent =
    'JevBench easy is 100% for every model and is omitted. typical-small-preview → typical-small is 75.0% → 69.4% on JevBench standard (about 1 SE at n = 72, SE ≈ 5.8 points), traded for typed heads and calibration (held-out score NLL 2.03 → 1.01).';
  container.appendChild(footnote);

  if (frozenByTrained.size) {
    const frozenNote = document.createElement('p');
    frozenNote.className = 'note';
    frozenNote.style.marginTop = '8px';
    frozenNote.textContent =
      'On the hard tier the frozen 4B (44.1%) beats the trained 4B (42.3%); training helps the standard tier at 1.7B (+17 points) far more than at 4B (+3).';
    container.appendChild(frozenNote);
  }
}

function uniqueSources(list) {
  return [...new Set(list.filter(Boolean))].join(' · ');
}

function caption(el, text) {
  if (!el) return;
  const p = document.createElement('p');
  p.className = 'note';
  p.textContent = text;
  el.appendChild(p);
}

function buildInTrainingBox(container, frozenDoc) {
  if (!container || !frozenDoc?.in_flight) return;
  container.innerHTML = '';
  const eyebrow = document.createElement('p');
  eyebrow.className = 't-eyebrow muted';
  eyebrow.style.marginBottom = '8px';
  eyebrow.textContent = 'In training';
  container.appendChild(eyebrow);
  const body = document.createElement('p');
  body.className = 'note';
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
      fmt: fmtPct,
      refLines: chanceDoc ? [{ y: chanceDoc.std.acc, label: `chance ${fmtPct(chanceDoc.std.acc)}` }] : [],
    });
    const chanceTxt = chanceDoc
      ? ` Chance baseline: ${fmtPct(chanceDoc.std.acc)} standard / ${fmtPct(chanceDoc.easy.acc)} easy / ${fmtPct(chanceDoc.hard.acc)} hard, n=${chanceDoc.std.n} (${chanceDoc.source}).`
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
      h4.className = 't-eyebrow muted';
      h4.style.marginBottom = '8px';
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
  // #tryit (the nav's CTA) points at a card that starts collapsed: open it on arrival, or the
  // visitor lands on a one-line summary and a Run button and has to guess.
  const openHash = () => {
    const target = location.hash.length > 1 ? document.querySelector(location.hash) : null;
    const card = target && (target.matches('[data-exhibit-card]') ? target : target.querySelector('[data-exhibit-card]')); // null, never false: ?. does not short-circuit on false
    if (card?.querySelector('[data-exhibit-body]')?.hidden) card.querySelector('[data-exhibit-toggle]')?.click();
  };
  addEventListener('hashchange', openHash);
  openHash();
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

// ---- hero decision card: cycles through real recorded decisions (presets are the replay keys) ----

function heroPool(presets, replays) {
  const pool = [];
  const add = (state, queries) => {
    const e = replays[hashKey(state, queries)];
    if (!e) return;
    queries.forEach((q, k) => { if (e.results[k] && q.labels.length <= 6) pool.push({ state, q, r: e.results[k], ms: e.ms, device: e.device, model: e.model }); });
  };
  if (presets.playground) { add(presets.playground.state, presets.playground.queries); presets.playground.queries.forEach((q) => add(presets.playground.state, [q])); }
  if (presets.policy) add(presets.policy.state, presets.policy.queries);
  if (presets.abstain) presets.abstain.variants.forEach((v) => add(presets.abstain.state, v.queries));
  if (presets.flip) presets.flip.orders.forEach((order) => {
    const crit = order.map((l) => `${l}: ${presets.flip.rules[l]}`).join('  ');
    add(presets.flip.state, [{ type: 'choice', question: `${presets.flip.question_prefix}\n${crit}`, labels: order }]);
  });
  return pool;
}

let heroLive = null; // set by pushDecision in live mode
async function mountHero(presets) {
  const read = document.getElementById('hero-decision');
  const scene = document.getElementById('hero-decision-scene');
  if (!read || !presets) return;
  const replays = (await loadJSON('data/replays.json')) || {};
  const pool = heroPool(presets, replays);
  if (!pool.length) return;
  // scene: the state as a typographic still; read: question + bars + meta
  scene.innerHTML = '';
  scene.style.cssText = 'padding:60px 50px;display:flex;flex-direction:column;justify-content:flex-end;gap:18px';
  const stateEl = document.createElement('p'); stateEl.className = 't-card'; stateEl.style.maxWidth = '22ch';
  const stateLab = document.createElement('p'); stateLab.className = 't-eyebrow muted'; stateLab.textContent = 'the state';
  scene.append(stateLab, stateEl);
  read.innerHTML = '';
  const qLab = document.createElement('p'); qLab.className = 't-eyebrow muted'; qLab.textContent = 'the question';
  const qEl = document.createElement('p'); qEl.className = 't-ui'; qEl.style.marginBottom = '18px';
  const barsEl = document.createElement('div');
  const meta = document.createElement('p'); meta.className = 't-mono muted'; meta.style.marginTop = '18px';
  const gate = document.createElement('p'); gate.className = 't-mono muted';
  gate.textContent = glued('candidate bars sum to one · the dashed ∅ row is a separate gate (p that none apply), not part of that sum');
  read.append(qLab, qEl, barsEl, meta, gate);
  const b = bars(barsEl, rowsFromResult(pool[0].r));
  const trunc = (t, n) => (t.length > n ? t.slice(0, n - 1) + '…' : t);
  const show = (e, source) => {
    stateEl.textContent = trunc(e.state, 220);
    qEl.textContent = trunc(e.q.question, 170);
    b.update(rowsFromResult(e.r));
    meta.textContent = glued(`${e.q.type} · ${source} · ${Math.round(e.ms)} ms · ${e.device || 'mps'} · ${e.model || 'typical-small'} · ${pool.length} recorded decisions`);
  };
  let i = 0;
  show(pool[0], 'recorded');
  // 8 s, not 3.2: at the old pace the state paragraph changed while it was being read. Hovering
  // or focusing the panel holds the current decision; a click advances it immediately.
  let hold = false;
  const panel = read.closest('.media') || read;
  panel.addEventListener('pointerenter', () => { hold = true; });
  panel.addEventListener('pointerleave', () => { hold = false; });
  const next = () => {
    if (heroLive) { show(heroLive, 'live'); heroLive = null; return; }
    i = (i + 1) % pool.length;
    show(pool[i], 'recorded');
  };
  panel.addEventListener('click', next);
  setInterval(() => { if (!hold) next(); }, 8000);
}
// live decide() results flow into the decision panel
function pushDecision(res, state, queries) {
  if (!res?.results?.[0] || !queries?.[0] || queries[0].labels.length > 6 || !state) return;
  heroLive = { state: String(state), q: queries[0], r: res.results[0], ms: res.ms, device: res.device, model: res.model };
}

// The hero film is shown as a character field, not a picture (js/ascii.js). Mounting is
// best-effort: no iframe, no canvas or an unreadable buffer and the film just plays as itself.
function mountFilmAscii(film) {
  const stage = film?.closest('.stage');
  if (!film || !stage) return;
  import('./ascii.js').then(({ mountAscii }) => {
    mountAscii(stage, () => {
      const c = film.contentDocument?.getElementById('canvas');
      return c && c.width ? c : null;
    }, { cols: 112, after: film });
  }).catch(() => {});
}

// ---- the film: real DOOM in the hero, driven by the model (live) or the rule list (recorded) ----
function mountFilm(ctx) {
  const film = document.getElementById('film');
  mountFilmAscii(film);
  const rowsEl = document.getElementById('hud-rows'), sentEl = document.getElementById('hud-sentence'), rec = document.getElementById('hud-rec');
  if (!film || !rowsEl) return;
  const labels = ['retreat', 'shoot', 'turn left', 'turn right', 'explore'];
  let stopped = false;
  const io = new IntersectionObserver((es) => { stopped = !es[0].isIntersecting; });
  io.observe(film);
  async function loop() {
    if (stopped) return setTimeout(loop, 800); // do not drive the game (or the model) while the hero is off screen
    const D = film.contentWindow?.Doom;
    if (!D || !D.ready) return setTimeout(loop, 500);
    const st = D.state();
    if (!st.in_level) return setTimeout(loop, 500);
    const cands = candidatesFor(st);
    const sentence = describeDoom(st);
    let move = scriptedPolicy(st), probs = null, source = 'rule list';
    if (ctx.mode() === 'live' && cands.length >= 2) {
      try {
        const res = await decide(sentence, [{ type: 'choice', question: DOOM_QUESTION, labels: cands }]);
        const r = res?.results?.[0];
        if (r) { probs = r.probs; move = cands.reduce((a, c) => ((r.probs[c] ?? 0) > (r.probs[a] ?? 0) ? c : a), cands[0]); source = `model · ${Math.round(res.ms)} ms on ${res.device || 'mps'}`; }
      } catch {}
    }
    rec.textContent = glued(`${source.startsWith('model') ? 'LIVE' : 'REC'} · typical-small · E1M1 · ${source}`);
    sentEl.textContent = sentence;
    rowsEl.innerHTML = '';
    labels.forEach((l) => {
      const d = document.createElement('div'); d.className = 'line' + (l === move ? ' on' : '');
      const a = document.createElement('span'); a.textContent = l;
      const v = document.createElement('span'); v.textContent = probs ? (probs[l] != null ? probs[l].toFixed(2) : '') : cands.includes(l) ? (l === move ? '●' : '·') : '';
      d.append(a, v); rowsEl.appendChild(d);
    });
    const action = resolveIntent(st, move);
    // The decision and the keystroke are not the same thing: "shoot" at nine cells means centre
    // the target and walk in first. Print both so the film doesn't look like it fires blindly.
    if (action !== move) sentEl.textContent = `${sentence}  →  ${move} · engine: ${action}`;
    const spec = keyPress(st, move, action);
    try { if (typeof spec[1] === 'object') await D.turnBy(spec[0], spec[1].deg); else await D.press(spec[0], spec[1]); } catch {}
    setTimeout(loop, 380);
  }
  resetNav(); loop();
}

async function mountFindings() {
  const el = document.getElementById('findings-cards');
  const doc = await loadJSON('data/findings.json');
  if (!el || !doc) return;
  doc.items.forEach((f) => {
    const card = document.createElement('div');
    card.className = 'card';
    const tag = document.createElement('span'); tag.className = 'mono'; tag.textContent = f.tag;
    const h = document.createElement('h3'); h.textContent = f.title;
    const p = document.createElement('p'); p.textContent = f.meta;
    card.append(tag, h, p);
    el.appendChild(card);
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
  glueSeparators(); // static prose, before anything awaits
  await probeHealth();

  const [presets, models, reliabilityDoc, chanceDoc, frozenDoc] = await Promise.all([
    loadJSON('data/presets.json'),
    loadJSON('data/models.json'),
    loadJSON('data/reliability.json'),
    loadJSON('data/chance.json'),
    loadJSON('data/frozen.json'),
  ]);

  const liveDecide = async (state, queries, opts) => { const res = await decide(state, queries, opts); pushDecision(res, state, queries); return res; };
  const ctx = { decide: liveDecide, mode, readout, presets, inkBars };

  mountHero(presets);
  mountFilm(ctx);
  import('./motion.js').then((m) => { const go = () => m.mountMotion(); if (window.gsap) go(); else window.addEventListener('load', go); });
  mountFindings();
  mountResults(models, reliabilityDoc, chanceDoc, frozenDoc);
  mountTryit(ctx);
  wireExhibits(ctx);
  glueSeparators();
  setTimeout(glueSeparators, 800); // the table, charts and captions mount async
  // The exhibit tags say what this visitor will actually get: every one of them decides live
  // against a reachable server and replays a recorded run otherwise, so the markup ships the
  // pessimistic label and only the live case upgrades it.
  if (mode() === 'live') document.querySelectorAll('.exrow-head .tag').forEach((t) => { t.textContent = 'Live'; });
  mountGameScreens(ctx);
}

boot();
