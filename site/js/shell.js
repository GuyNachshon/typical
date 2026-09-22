// Page shell for v4 ("Atoms"): mode probe, hero mark mosaic, results table + charts, the
// frozen-backbone control, exhibit expand/collapse (mount on first expand), and the lazy-mount
// loader for js/games/render-<name>.js. Static, placed, not kinetic — no GSAP, no reveals.
// Every number that reaches the page comes from data/*.json (precompute output) or a live
// decide() call — never an invented one.
import { decide, mode, probeHealth } from './api.js';
import { bars, inkBars, rowsFromResult } from './bars.js';
import { glueSeparators, glued, bindWidows } from './typography.js';
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
    a.textContent = 'weights\u00A0→';
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
      td(m.id.replace(/-/g, '\u2011')), // a model id is one token; plain hyphens let it split across lines
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
      td(m.id.replace(/-/g, '\u2011')), // a model id is one token; plain hyphens let it split across lines
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
  p.textContent = glued(text); // captions mount after the document-wide pass has run
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
  const host = document.getElementById('hero-decision');
  if (!host || !presets?.playground) return;
  const replays = (await loadJSON('data/replays.json')) || {};
  const pg = presets.playground;
  // One state, its four typed questions. Each is recorded on its own (scripts/record_replays.py)
  // so the instrument cycles even with no server, and upgrades itself to live when there is one.
  const entries = pg.queries
    .map((q) => ({ q, result: replays[hashKey(pg.state, [q])]?.results?.[0], ms: 0 }))
    .filter((e) => e.result)
    .map((e) => ({ ...e, result: { ...e.result, ms: replays[hashKey(pg.state, [e.q])]?.ms ?? 0 } }));
  if (!entries.length) return;
  const { mountInstrument } = await import('./instrument.js');
  mountInstrument(host, {
    state: pg.state,
    entries,
    decide: mode() === 'live' ? decide : null,
    onLive: pushDecision,
  });
}
// live decide() results flow into the decision panel
function pushDecision(res, state, queries) {
  if (!res?.results?.[0] || !queries?.[0] || queries[0].labels.length > 6 || !state) return;
  heroLive = { state: String(state), q: queries[0], r: res.results[0], ms: res.ms, device: res.device, model: res.model };
}

// The hero film is shown as a character field, not a picture (js/ascii.js). Mounting is
// best-effort: no iframe, no canvas or an unreadable buffer and the film just plays as itself.
// Returns a handle whose pulse() the decision loop calls, or a no-op if the effect never mounts.
function mountFilmFxOn(film, onReady) {
  const stage = film?.closest('.stage');
  let heldSince = 0; // when the level last became live and stayed that way
  const handle = { pulse() {}, stop() {} };
  if (!film || !stage) {
    onReady?.();
    return handle;
  }
  import('./filmfx.js').then(({ mountFilmFx }) => {
    const fx = mountFilmFx(stage, () => {
      const c = film.contentDocument?.getElementById('canvas');
      return c && c.width ? c : null;
    }, {
      after: film,
      onReady,
      // in_level alone is not enough: DOOM's attract mode plays a recorded demo, and that reports
      // in_level too, so the cold start handed over to the title screen. Start the level ourselves
      // and only reveal once that game has been running for a moment.
      // The level has to be *settled*, not merely reported once. DOOM drops back to its title
      // screen on its own, and its attract-mode demo reports in_level too, so a single true
      // reading handed the switch over to whatever happened to be on screen. Require the level
      // to hold continuously, and restart it if it slips back to the title.
      // Is the game actually showing play right now? The treatment holds its last frame when not.
      live: () => {
        const D = film.contentWindow?.Doom;
        if (!D?.ready) return false;
        try {
          const st = D.state();
          return !!(st.in_level && (st.health ?? 0) > 0);
        } catch {
          return false;
        }
      },
      // Observes only. Starting the level is the film loop's job and nobody else's: when this
      // also called newGame, the two of them raced and the engine bounced between a fresh level
      // and its own title screen, which is what the switch kept landing on.
      ready: () => {
        const D = film.contentWindow?.Doom;
        if (!D?.ready) return false;
        try {
          const st = D.state();
          if (!(st.in_level && (st.health ?? 0) > 0)) {
            heldSince = 0;
            return false;
          }
          if (!heldSince) heldSince = performance.now();
          return performance.now() - heldSince > 1600;
        } catch {
          return false;
        }
      },
    });
    handle.pulse = fx.pulse;
    handle.stop = fx.stop;
    handle.grade = fx.grade;
    handle.phase = fx.phase;
  }).catch(() => onReady?.());
  return handle;
}

// ---- the film: real DOOM in the hero, driven by the model (live) or the rule list (recorded) ----
function mountFilm(ctx, ids = {}) {
  const film = document.getElementById(ids.film || 'film');
  // The treatment is the hero's alone: the card in chapter 04 shows the frame untouched.
  // While the tube is running its cold start the page holds back: nav, copy and the probability
  // panel stay hidden so the first thing a visitor reads is what the model is.
  // The cards arrive a beat after the picture does, not with it: the switch should land on the
  // game alone, and only then does the page assemble itself around it.
  const fx = ids.fx === true
    ? mountFilmFxOn(film, () => setTimeout(() => document.documentElement.classList.remove('booting'), 700))
    : { pulse() {} };
  if (ids.fx === true && typeof window !== 'undefined') { window.__fxPulse = () => fx.pulse(); window.__fxGrade = () => fx.grade?.(); window.__fxPhase = () => fx.phase?.(); } // probe hooks for the effect check
  const rowsEl = document.getElementById(ids.rows || 'hud-rows');
  const sentEl = document.getElementById(ids.sentence || 'hud-sentence');
  const rec = document.getElementById(ids.rec || 'hud-rec');
  const rec2 = document.getElementById(ids.rec2 || (ids.rec ? `${ids.rec}2` : 'hud-rec2'));
  if (!film || !rowsEl) return;
  // Each film owns its own navigator state; two instances sharing the module-level route counter
  // would walk each other's waypoints.
  const labels = ['retreat', 'shoot', 'turn left', 'turn right', 'explore'];
  let stopped = false;
  const io = new IntersectionObserver((es) => { stopped = !es[0].isIntersecting; });
  io.observe(film);
  // The run has to survive itself: a player that walks into nukage loses health every tick, and
  // "badly hurt → retreat" against a wall is a loop that ends in a corpse on the hero of the
  // page. Restart when the player dies, leaves the level, or stops making progress while taking
  // damage — and start the route again from the top.
  // Two stages, because most stalls are not fatal: a player wedged against a wall takes no damage
  // and would otherwise stand there for ever (which is also what draws the smeared "hall of
  // mirrors" frame — DOOM renders that when the view is inside geometry). Nudge first, restart
  // only if the nudge fails.
  const NUDGE_TICKS = 8; // ~6 s of no progress -> turn hard and walk
  const STUCK_TICKS = 24; // ~18 s -> give up and restart the level
  let anchor = null;
  let stuckFor = 0;
  let lastHealth = 100;
  let lastRestart = 0;
  function restart(D) {
    // A cooldown matters: newGame while a previous start is still being processed leaves the
    // engine flipping between the level and the title screen.
    const now = performance.now();
    if (now - lastRestart < 900) return;
    lastRestart = now;
    stuckFor = 0;
    anchor = null;
    lastHealth = 100;
    resetNav();
    try { D.newGame(3); } catch {}
  }

  async function loop() {
    if (stopped) return setTimeout(loop, 800); // do not drive the game (or the model) while the hero is off screen
    const D = film.contentWindow?.Doom;
    if (!D || !D.ready) return setTimeout(loop, 500);
    const st = D.state();
    if (!st.in_level || (st.health ?? 100) <= 0) {
      restart(D); // also how the very first level gets started: the harness no longer does it
      return setTimeout(loop, 700);
    }
    if (!anchor || Math.hypot(st.x - anchor.x, st.y - anchor.y) > 90) {
      anchor = { x: st.x, y: st.y };
      stuckFor = 0;
    } else {
      stuckFor += 1;
    }
    lastHealth = st.health ?? lastHealth;
    if (typeof window !== 'undefined' && window.__filmProbe) window.__filmStuck = stuckFor;
    if (stuckFor > STUCK_TICKS) {
      restart(D);
      return setTimeout(loop, 1500);
    }
    if (stuckFor > 0 && stuckFor % NUDGE_TICKS === 0) {
      // hard turn + a walk, outside the model's decision: the engine is getting the player out of
      // a corner, not choosing an action, so the HUD keeps showing the model's own last call
      try {
        await D.turnBy(stuckFor % (NUDGE_TICKS * 2) === 0 ? 'right' : 'left', 100);
        await D.press('forward', 650);
      } catch {}
      return setTimeout(loop, 120);
    }
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
    // Two lines by construction: one wrapped line used to start with a dangling separator.
    rec.textContent = `${source.startsWith('model') ? 'LIVE' : 'REC'} · typical-small · E1M1`;
    if (rec2) rec2.textContent = source;
    sentEl.textContent = sentence;
    rowsEl.innerHTML = '';
    fx.pulse(); // the probability panel is being rewritten: run a processing pulse through the film
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
  bindWidows();
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
  // set before the film mounts; filmfx clears it the moment the cold start is over (or at once,
  // if it is skipped for a repeat visit or for reduced motion)
  document.documentElement.classList.add('booting');
  mountFilm(ctx, { fx: true });
  // The same game again in chapter 04, plain: no bloom layer, so the card shows the frame exactly
  // as the engine draws it. Only one of the two runs at a time — each pauses when off screen.
  mountFilm(ctx, { film: 'film-card', rows: 'card-rows', sentence: 'card-sentence', rec: 'card-rec' });
  import('./motion.js').then((m) => {
    const go = () => {
      if (document.documentElement.classList.contains('booting')) {
        // wait for the cold start rather than playing the hero entrance behind it
        const obs = new MutationObserver(() => {
          if (!document.documentElement.classList.contains('booting')) {
            obs.disconnect();
            m.mountMotion();
          }
        });
        obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
        setTimeout(() => { obs.disconnect(); m.mountMotion(); }, 12000); // never strand the page
        return;
      }
      m.mountMotion();
    };
    if (window.gsap) go(); else window.addEventListener('load', go);
  });
  mountFindings();
  mountResults(models, reliabilityDoc, chanceDoc, frozenDoc);
  mountTryit(ctx);
  wireExhibits(ctx);
  glueSeparators();
  bindWidows();
  setTimeout(() => { glueSeparators(); bindWidows(); }, 800); // the table, charts and captions mount async
  // The exhibit tags say what this visitor will actually get: every one of them decides live
  // against a reachable server and replays a recorded run otherwise, so the markup ships the
  // pessimistic label and only the live case upgrades it.
  if (mode() === 'live') document.querySelectorAll('.exrow-head .tag').forEach((t) => { t.textContent = 'Live'; });
  mountGameScreens(ctx);
}

boot();
