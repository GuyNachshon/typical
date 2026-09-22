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
import { costBar } from './charts.js';

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

// ---- the scoreboard ----------------------------------------------------------------
//
// This replaced two tables of thirteen columns, a scaling ladder, four reliability histograms and
// a paragraph of lab notes. None of that was wrong; all of it was written for a reviewer, and a
// reviewer has the research page. What a reader needs from a results section is a number and
// something to measure it against, so every row carries its own baseline: guessing at 150 intents
// gets you 0.7%, the majority class on PagerDuty gets you 79.2%. Without the second number the
// first one means nothing, which is why "80.4%" was the least informative figure on the page.
//
// The last row is the one both models fail. It stays on the board, marked, because a scoreboard
// that only lists wins is an advertisement.
const SCORE_ROWS = [
  {
    task: 'Route to one of 150 intents you define',
    bench: 'CLINC-150',
    get: (m) => m.topic_intent.clinc,
    base: 1 / 150,
    baseLabel: 'guessing: 0.7%',
  },
  {
    task: 'Triage real incident tickets',
    bench: 'PagerDuty',
    get: (m) => m.external.pagerduty,
    base: 0.792,
    baseLabel: 'always guess the commonest: 79.2%',
  },
  {
    task: 'Answer a bounded workflow question',
    bench: 'JevBench standard',
    get: (m) => m.jevbench.std.acc,
    base: 0.311,
    baseLabel: 'guessing: 31.1%',
  },
];

function scoreCell(value, base, miss) {
  const cell = document.createElement('div');
  cell.className = `score-cell${miss ? ' is-miss' : ''}`;
  const num = document.createElement('p');
  num.className = 'score-num';
  num.textContent = pct(value);
  const track = document.createElement('div');
  track.className = 'score-track';
  const fill = document.createElement('i');
  fill.style.width = `${(value * 100).toFixed(1)}%`;
  // the baseline as a mark on the same track: the gap between the rule and the tick is the result
  const tick = document.createElement('b');
  tick.style.left = `${(base * 100).toFixed(1)}%`;
  track.append(fill, tick);
  cell.append(num, track);
  return cell;
}

function buildResultsTable(container, models) {
  const released = models.filter((m) => m.released && !m.id.includes('preview'));
  if (!released.length) return;
  const board = document.createElement('div');
  board.className = 'score';

  const head = document.createElement('div');
  head.className = 'score-row is-head';
  head.appendChild(document.createElement('div'));
  released.forEach((m) => {
    const h = document.createElement('p');
    h.className = 'score-model t-mono';
    h.textContent = `${m.id.replace(/-/g, '\u2011')} · ${m.params}`;
    head.appendChild(h);
  });
  board.appendChild(head);

  SCORE_ROWS.forEach((row) => {
    const tr = document.createElement('div');
    tr.className = 'score-row';
    const label = document.createElement('div');
    label.className = 'score-label';
    const task = document.createElement('p');
    task.className = 'score-task';
    task.textContent = row.task;
    const sub = document.createElement('p');
    sub.className = 'score-sub t-mono';
    sub.textContent = `${row.bench} · ${row.baseLabel}`;
    label.append(task, sub);
    if (row.miss) {
      const flag = document.createElement('p');
      flag.className = 'score-flag';
      flag.textContent = row.miss;
      label.appendChild(flag);
    }
    tr.appendChild(label);
    released.forEach((m) => tr.appendChild(scoreCell(row.get(m), row.base, row.miss)));
    board.appendChild(tr);
  });

  container.appendChild(board);
  const key = document.createElement('p');
  key.className = 'note mt-18';
  key.textContent = glued('The tick on each bar is the baseline for that task — what you get without a model at all.');
  container.appendChild(key);
}

// ---- the comparison: the same benchmark, run against models you could use instead ------------
//
// frozen.json is the general-purpose backbones prompted three-shot over the rendered options, on
// the same 231 public ids as our own run. That is the only comparison on this site that is
// apples-to-apples, which is why it is the only one here: same benchmark, same items, same
// protocol, and the sizes are on the chart because size is the trade being made.
function paramsOf(name) {
  const m = /(\d+(?:\.\d+)?)B/.exec(name);
  return m ? Number(m[1]) : null;
}

function buildComparison(container, models, frozenDoc) {
  if (!container || !frozenDoc?.rows?.length) return;
  const ours = models
    .filter((m) => m.released && !m.id.includes('preview'))
    .map((m) => ({ name: m.id.replace(/-/g, '\u2011'), note: 'decision model', acc: m.jevbench.std.acc, params: paramsOf(m.params), ours: true }));
  const theirs = frozenDoc.rows.map((r) => ({ name: r.backbone, note: 'prompted, 3 exemplars', acc: r.std, params: paramsOf(r.backbone) }));
  const rows = [...ours, ...theirs].sort((a, b) => b.acc - a.acc);

  const list = document.createElement('div');
  list.className = 'rank';
  rows.forEach((r) => {
    const row = document.createElement('div');
    row.className = `rank-row${r.ours ? ' is-ours' : ''}`;
    const label = document.createElement('div');
    label.className = 'rank-label';
    const n = document.createElement('p');
    n.className = 'rank-name';
    n.textContent = r.name;
    const sub = document.createElement('p');
    sub.className = 'rank-sub t-mono';
    sub.textContent = r.note;
    label.append(n, sub);
    const size = document.createElement('p');
    size.className = 'rank-size t-mono';
    size.textContent = r.params ? `${r.params}B` : '';
    const track = document.createElement('div');
    track.className = 'rank-track';
    const fill = document.createElement('i');
    // the full 0-100% scale, not normalised to the best row: dividing by the leader made 81.9%
    // draw as a full bar and flattened every difference underneath it
    fill.style.width = `${(r.acc * 100).toFixed(1)}%`;
    const tick = document.createElement('b'); // chance, so every bar is read against the same floor
    tick.style.left = '31.1%';
    track.append(fill, tick);
    const val = document.createElement('p');
    val.className = 'rank-val';
    val.textContent = pct(r.acc);
    row.append(label, size, track, val);
    list.appendChild(row);
  });
  container.replaceChildren(list);

  const note = document.createElement('p');
  note.className = 'note mt-18';
  note.textContent = glued(
    'JevBench standard, the same 231 public ids for every row; the tick on each bar is chance, 31.1%. The general models read the options as ' +
      'rendered text and answer with a letter, three exemplars in front of them; ours read the same state ' +
      'and return a distribution. At n = 72 one standard error is about 5.8 points, so treat anything ' +
      'inside six points as a tie \u2014 including the gap at 4B.'
  );
  container.appendChild(note);
}

function caption(el, text) {
  if (!el) return;
  const p = document.createElement('p');
  p.className = 'note';
  p.textContent = glued(text); // captions mount after the document-wide pass has run
  el.appendChild(p);
}

async function mountResults(models, reliabilityDoc, chanceDoc, frozenDoc) {
  // reliabilityDoc and chanceDoc are the research page's business now; kept in the signature so
  // the one caller does not have to change shape
  const tableEl = document.getElementById('results-table');
  if (!models) return;
  if (tableEl) buildResultsTable(tableEl, models);
  buildComparison(document.getElementById('results-compare'), models, frozenDoc);

  const g2 = document.getElementById('chart-g2');
  if (g2) {
    // Released models only. The 14B is not out, and a fourth near-identical line was the reason
    // this chart said nothing — the point is where the time goes, not which checkpoint wins by 3 ms.
    costBar(g2, {
      m: 32,
      // the preview borrows typical-small's ladder (latency.note), so plotting it would draw the
      // same bar twice under two names
      rows: models.filter((m) => m.released && m.latency.marginal_ms_m32 && !m.latency.note).map((m) => ({
        model: m.id.replace(/-/g, '\u2011'),
        one: m.latency.single_ms.k2,
        marginal: m.latency.marginal_ms_m32.k2,
      })),
      title: 'one decision, and thirty-two on the same state',
    });
    // A repo path is not a source a reader can check; the conditions are what the number means.
    caption(g2.parentElement, 'Measured on one H100, one decision at a time, a 256\u2011token state and two options per question, model load excluded. A wider answer space costs more: at 256 options a single decision is 106 ms and each further question 28. Full ladder on the research page.');
  }

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
// One state, three questions, resolving together (js/primitives.js).
async function mountPrimitives(presets) {
  const host = document.getElementById('primitives');
  if (!host || !presets?.playground) return;
  const replays = (await loadJSON('data/replays.json')) || {};
  const pg = presets.playground;
  const recorded = (q) => {
    const hit = replays[hashKey(pg.state, [q])];
    const r = hit?.results?.[0];
    return r ? { ...r, ms: hit.ms ?? 0 } : undefined;
  };
  const byType = (type) => pg.queries.find((q) => q.type === type);
  const questions = ['choice', 'noul', 'score'].map(byType).filter(Boolean);
  if (questions.length !== 3 || !questions.every((q) => recorded(q))) return;
  const { mountProgram } = await import('./primitives.js');
  mountProgram(host, {
    state: pg.state,
    questions,
    resultFor: recorded,
    decide: mode() === 'live' ? decide : null,
  });
}

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
    ? mountFilmFxOn(film, () => document.documentElement.classList.remove('booting'))
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
  mountPrimitives(presets);
  // set before the film mounts; filmfx clears it the moment the cold start is over (or at once,
  // if it is skipped for a repeat visit or for reduced motion)
  mountFilm(ctx, { fx: true });
  import('./motion.js').then((m) => {
    const go = () => {
      if (!document.documentElement.classList.contains('booting')) {
        m.mountMotion();
        return;
      }
      // wait for the cold start rather than playing the hero entrance behind it, but never strand
      // the page if it never finishes. Whichever path gets there first cancels the other: an
      // uncancelled fallback fires twelve seconds later and runs the choreography a second time
      // over a page that already played it, which re-hides every chapter head.
      let fallback = 0;
      const obs = new MutationObserver(() => {
        if (document.documentElement.classList.contains('booting')) return;
        obs.disconnect();
        clearTimeout(fallback);
        m.mountMotion();
      });
      fallback = setTimeout(() => { obs.disconnect(); m.mountMotion(); }, 12000);
      obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
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
