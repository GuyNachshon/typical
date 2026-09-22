// explorables.js — P1-P4 interactive pieces (designs/explainer.md). Plain DOM + hand-rolled SVG,
// no libraries. Reuses charts.js (scale/niceTicks/reliability) and bars.js (the candidate-bar
// readout) where the shape matches; the scrub rail and the two run-explorer pivots are new SVG,
// same v11 ink/mid-gray/steel vocabulary (see charts.js header comment).
import { scale, niceTicks, reliability, fmtNum } from './charts.js';
import { bars } from './bars.js';

const INK = '#292827';
const MID = '#938f89';
const STEEL = '#c2bfba';
const NS = 'http://www.w3.org/2000/svg';

function isNum(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

function svgEl(tag, attrs = {}) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

async function loadJSON(path) {
  try {
    const res = await fetch(path);
    if (res.ok) return await res.json();
  } catch {
    // static file:// serving without a data/ dir - explorable renders its error state below
  }
  return null;
}

function skeleton(el, path) {
  el.innerHTML = '';
  const p = document.createElement('p');
  p.className = 'note explorable-skeleton';
  p.textContent = `···· loading ${path}`;
  el.appendChild(p);
}

function missing(el, path) {
  el.innerHTML = '';
  const p = document.createElement('p');
  p.className = 'note';
  p.textContent = `${path} missing; chart withheld`;
  el.appendChild(p);
}

function sourceLine(el, text) {
  const p = document.createElement('div');
  p.className = 'chart-source';
  p.textContent = `source: ${text}`;
  el.appendChild(p);
}

// Segmented .btn.ghost control - the one interaction primitive every piece steps through.
function seg(options, value, onChange, ariaLabel) {
  const wrap = document.createElement('div');
  wrap.className = 'seg';
  wrap.setAttribute('role', 'group');
  if (ariaLabel) wrap.setAttribute('aria-label', ariaLabel);
  const buttons = options.map((opt) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'btn ghost';
    b.textContent = opt.label ?? String(opt.value);
    b.setAttribute('aria-pressed', String(opt.value === value));
    b.addEventListener('click', () => {
      if (b.getAttribute('aria-pressed') === 'true') return;
      buttons.forEach((o) => o.setAttribute('aria-pressed', 'false'));
      b.setAttribute('aria-pressed', 'true');
      onChange(opt.value);
    });
    wrap.appendChild(b);
    return b;
  });
  return wrap;
}

function checks(options, active, onChange, ariaLabel) {
  const wrap = document.createElement('div');
  wrap.className = 'seg';
  wrap.setAttribute('role', 'group');
  if (ariaLabel) wrap.setAttribute('aria-label', ariaLabel);
  options.forEach((opt) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'btn ghost';
    b.textContent = opt.label ?? String(opt.value);
    b.setAttribute('aria-pressed', String(active.has(opt.value)));
    b.addEventListener('click', () => {
      if (active.has(opt.value)) active.delete(opt.value); else active.add(opt.value);
      b.setAttribute('aria-pressed', String(active.has(opt.value)));
      onChange(active);
    });
    wrap.appendChild(b);
  });
  return wrap;
}

// ============================================================================
// P1 — latency explorer (native L x K x M grid)
// ============================================================================
async function mountLatency() {
  const el = document.getElementById('chart-latency');
  if (!el) return;
  skeleton(el, 'data/latency.json');
  const d = await loadJSON('data/latency.json');
  if (!d) return missing(el, 'data/latency.json');

  const Ls = Object.keys(d.native_sweep).map(Number).sort((a, b) => a - b);
  const Ks = [2, 4, 10, 32, 64, 128, 256];
  const Ms = [1, 32, 256];
  const state = { L: 256, K: 32, M: 1 };

  el.innerHTML = '';
  const root = document.createElement('div');
  root.className = 'explorable explorable-latency';
  el.appendChild(root);

  const controls = document.createElement('div');
  controls.className = 'explorable-controls';
  root.appendChild(controls);

  const rowState = document.createElement('div');
  rowState.className = 'explorable-row';
  rowState.append(labelSpan('state tokens'), seg(Ls.map((v) => ({ value: v, label: String(v) })), state.L, (v) => { state.L = v; render(); }, 'state length'));
  const rowK = document.createElement('div');
  rowK.className = 'explorable-row';
  rowK.append(labelSpan('candidates K'), seg(Ks.map((v) => ({ value: v, label: String(v) })), state.K, (v) => { state.K = v; render(); }, 'candidates K'));
  const rowM = document.createElement('div');
  rowM.className = 'explorable-row';
  rowM.append(labelSpan('questions M'), seg(Ms.map((v) => ({ value: v, label: String(v) })), state.M, (v) => { state.M = v; render(); }, 'questions M'));
  controls.append(rowState, rowK, rowM);

  const body = document.createElement('div');
  body.className = 'explorable-body';
  root.appendChild(body);

  function labelSpan(text) {
    const s = document.createElement('span');
    s.className = 't-mono muted explorable-label';
    s.textContent = text;
    return s;
  }

  function barRow(label, ms, maxMs, extra) {
    const row = document.createElement('div');
    row.className = 'bar';
    row.style.gridTemplateColumns = 'minmax(96px, 150px) 1fr 64px'; // src.css's 44px value column fits "0.65", not "45.2 ms" - widen locally, src.css itself untouched
    const l = document.createElement('span');
    l.className = 'label';
    l.textContent = label;
    const track = document.createElement('div');
    track.className = 'track';
    const fill = document.createElement('div');
    fill.className = 'fill';
    fill.style.width = `${Math.max(0, Math.min(100, (ms / maxMs) * 100)).toFixed(1)}%`;
    track.title = extra;
    track.appendChild(fill);
    const v = document.createElement('span');
    v.className = 'v';
    v.textContent = `${ms.toFixed(1)} ms`;
    row.append(l, track, v);
    return row;
  }

  function render() {
    body.innerHTML = '';
    const byK = d.native_sweep[String(state.L)] || {};
    const cell = byK[String(state.K)];
    if (!cell) {
      const p = document.createElement('p');
      p.className = 'note';
      p.textContent = `not measured (OOM in baseline) — L=${state.L}, K=${state.K}`;
      body.appendChild(p);
      sourceLine(el.parentElement.contains(el) ? body : body, d.native_sweep_source || 'site/data/latency.json.native_sweep');
      return;
    }
    const nativeRow = cell.native.find((r) => r.m === state.M) || cell.native[0];
    const baseRow = cell.baseline.find((r) => r.m === state.M) || cell.baseline[0];
    const maxMs = Math.max(nativeRow.total_ms, baseRow.total_ms);

    const barsWrap = document.createElement('div');
    barsWrap.className = 'bars';
    barsWrap.appendChild(barRow('native', nativeRow.total_ms, maxMs,
      `state ${nativeRow.t_state_ms.toFixed(1)} ms + queries ${nativeRow.t_queries_ms.toFixed(1)} ms`));
    barsWrap.appendChild(barRow('prompting', baseRow.total_ms, maxMs,
      `state ${baseRow.t_state_ms.toFixed(1)} ms + queries ${baseRow.t_queries_ms.toFixed(1)} ms`));
    body.appendChild(barsWrap);

    const tiles = document.createElement('div');
    tiles.className = 'explorable-tiles';
    tiles.appendChild(tile('ms per extra question, state cached',
      `native ${nativeRow.marginal_ms.toFixed(1)} ms · prompting ${baseRow.marginal_ms.toFixed(1)} ms`));
    tiles.appendChild(tile('peak memory', `${(cell.peak_mb / 1024).toFixed(1)} GB`));
    const crossK = cell.crossover ? cell.crossover[String(state.M)] : null;
    tiles.appendChild(tile('crossover',
      crossK != null ? `native overtakes prompting at K ≥ ${crossK} for M=${state.M}` : 'not sourced for this M'));
    body.appendChild(tiles);

    const ladder = d.ladder['typical-small'];
    const note = document.createElement('p');
    note.className = 'note explorable-note';
    note.textContent = `Bench checkpoint: nc_n3 (research checkpoint, same shape as the release, not the release). `
      + `Release ladder — typical-small ${ladder.single_ms.k2}/${ladder.single_ms.k32}/${ladder.single_ms.k256} ms `
      + `(K=2/32/256, one decision), marginal at M=32: ${ladder.marginal_ms_m32.k2}/${ladder.marginal_ms_m32.k32}/${ladder.marginal_ms_m32.k256} ms — from latency.json.ladder, not live.`;
    body.appendChild(note);

    sourceLine(body, d.native_sweep_source || `site/data/latency.json.native_sweep.${state.L}.${state.K} + .ladder`);
  }

  function tile(title, value) {
    const t = document.createElement('div');
    t.className = 'explorable-tile';
    const h = document.createElement('p');
    h.className = 't-mono muted';
    h.textContent = title;
    const v = document.createElement('p');
    v.className = 't-feature';
    v.textContent = value;
    t.append(h, v);
    return t;
  }

  render();
}

// ============================================================================
// P2 — calibration explorer (model x tier, ECE arithmetic)
// ============================================================================
async function mountCalibration() {
  const el = document.getElementById('chart-calibration');
  if (!el) return;
  skeleton(el, 'data/reliability.json');
  const [rel, models, chance] = await Promise.all([
    loadJSON('data/reliability.json'), loadJSON('data/models.json'), loadJSON('data/chance.json'),
  ]);
  if (!rel || !models || !chance) return missing(el, 'data/reliability.json');

  const MODEL_OPTS = [
    { value: 'typical-small', label: 'small' },
    { value: 'typical-medium', label: 'medium' },
    { value: 'typical-small-preview', label: 'preview' },
  ];
  const state = { model: 'typical-small', tier: 'std' };

  el.innerHTML = '';
  const root = document.createElement('div');
  root.className = 'explorable explorable-calibration';
  el.appendChild(root);

  const controls = document.createElement('div');
  controls.className = 'explorable-row';
  controls.append(
    seg(MODEL_OPTS, state.model, (v) => { state.model = v; render(); }, 'model'),
    seg([{ value: 'std', label: 'standard' }, { value: 'hard', label: 'hard' }], state.tier, (v) => { state.tier = v; render(); }, 'tier'),
  );
  root.appendChild(controls);

  const chartHost = document.createElement('div');
  const arithHost = document.createElement('div');
  arithHost.className = 'calib-arith';
  root.append(chartHost, arithHost);

  function render() {
    const bins = (rel[state.model] || {})[state.tier] || [];
    const info = models.find((m) => m.id === state.model);
    const modelTier = state.tier === 'std' ? 'std' : 'hard';
    const published = info?.jevbench?.[modelTier]?.ece ?? null;
    const n = info?.jevbench?.[modelTier]?.n ?? bins.reduce((a, b) => a + b.n, 0);
    const chanceAcc = chance[modelTier]?.acc;

    // reliability() (charts.js, not ours to edit) formats accuracy/mean_confidence with
    // .toFixed unconditionally, which throws on the n=0 bins' real `null` values — pass a
    // display-only copy with 0 in place of null; our own arithmetic list below (which does
    // handle n=0) reads the original `bins`, so nothing here is misrepresented as measured.
    const chartBins = bins.map((b) => ({ ...b, accuracy: b.accuracy ?? 0, mean_confidence: b.mean_confidence ?? 0 }));
    reliability(chartHost, { bins: chartBins, ece: published, title: `${state.model} · ${state.tier} reliability` });
    // charts.js reliability() only draws <rect> for the bins themselves - safe to zip by index.
    const rects = [...chartHost.querySelectorAll('svg > g > rect')];
    const label = document.createElement('div');
    label.className = 'chart-hover-label';
    label.hidden = true;
    chartHost.appendChild(label);
    rects.forEach((rect, i) => {
      const b = bins[i];
      if (!b || !b.n) return;
      rect.addEventListener('mouseenter', () => {
        label.hidden = false;
        label.textContent = `[${b.lo.toFixed(2)}–${b.hi.toFixed(2)}] n ${b.n}/${n} × |${b.accuracy.toFixed(3)} − ${b.mean_confidence.toFixed(3)}| = ${((b.n / n) * Math.abs(b.accuracy - b.mean_confidence)).toFixed(3)}`;
      });
      rect.addEventListener('mouseleave', () => { label.hidden = true; });
    });

    arithHost.innerHTML = '';
    let sum = 0;
    const list = document.createElement('ol');
    list.className = 'calib-list';
    bins.forEach((b) => {
      const li = document.createElement('li');
      if (!b.n) {
        li.className = 'dim';
        li.textContent = `bin ${b.lo.toFixed(1)}–${b.hi.toFixed(1)}: n 0`;
      } else {
        const contrib = (b.n / n) * Math.abs(b.accuracy - b.mean_confidence);
        sum += contrib;
        li.innerHTML = `bin ${b.lo.toFixed(1)}–${b.hi.toFixed(1)}: n ${b.n}/${n} × |${b.accuracy.toFixed(3)} − ${b.mean_confidence.toFixed(3)}| = <b>${contrib.toFixed(3)}</b>`;
      }
      list.appendChild(li);
    });
    arithHost.appendChild(list);

    const sumLine = document.createElement('p');
    sumLine.className = 't-feature calib-sum';
    const diff = published != null ? Math.abs(sum - published) : null;
    sumLine.textContent = `sum = ${sum.toFixed(3)}` + (published != null ? ` · published ECE ${published.toFixed(3)}` : '')
      + (chanceAcc != null ? ` · chance ${chanceAcc.toFixed(3)}` : '');
    arithHost.appendChild(sumLine);
    if (diff != null && diff > 0.001) {
      const flag = document.createElement('p');
      flag.className = 'note';
      flag.textContent = `recomputed sum differs from the published ECE by ${diff.toFixed(3)} (> .001) — both numbers shown above, not reconciled.`;
      arithHost.appendChild(flag);
    }
    sourceLine(arithHost, `site/data/reliability.json.${state.model}.${state.tier} + site/data/models.json[${state.model}].jevbench.${modelTier}.ece + site/data/chance.json.${modelTier}`);
  }

  render();
}

// ============================================================================
// P3 — architecture trace ("trace one decision")
// ============================================================================
const STAGES = ['state', 'trunk', 'cache', 'suffix', 'head', 'probs'];
const STAGE_LABEL = { state: 'state tokens', trunk: 'trunk', cache: 'KV cache', suffix: 'suffix', head: 'head', probs: 'probabilities' };

async function mountTrace() {
  const el = document.getElementById('chart-trace');
  if (!el) return;
  skeleton(el, 'data/trace.json');
  const d = await loadJSON('data/trace.json');
  if (!d) return missing(el, 'data/trace.json');

  const QLABEL = { choice: 'team', score: 'urgency', noul: 'escalate' }; // 4th choice question ("want") disambiguated by index below
  const qLabels = d.questions.map((q, i) => (i === 0 ? 'team' : i === 3 ? 'want' : QLABEL[q.type] || q.type));

  const state = { stage: 0, q: 0 };

  el.innerHTML = '';
  const root = document.createElement('div');
  root.className = 'explorable explorable-trace';
  el.appendChild(root);

  const qRow = document.createElement('div');
  qRow.className = 'explorable-row';
  qRow.append(seg(qLabels.map((l, i) => ({ value: i, label: `${i + 1} ${l}` })), state.q, (v) => { state.q = v; render(); }, 'question'));
  root.appendChild(qRow);

  const rail = document.createElement('div');
  rail.className = 'scrub-rail';
  const ticks = document.createElement('div');
  ticks.className = 'scrub-ticks';
  STAGES.forEach((s, i) => {
    const t = document.createElement('button');
    t.type = 'button';
    t.className = 'scrub-tick';
    t.textContent = STAGE_LABEL[s];
    t.addEventListener('click', () => { state.stage = i; input.value = String(i); render(); });
    ticks.appendChild(t);
  });
  const input = document.createElement('input');
  input.type = 'range';
  input.className = 'scrub';
  input.min = '0';
  input.max = String(STAGES.length - 1);
  input.step = '1';
  input.value = '0';
  input.setAttribute('aria-label', 'trace stage');
  input.addEventListener('input', () => { state.stage = Number(input.value); render(); });
  const nav = document.createElement('div');
  nav.className = 'scrub-nav';
  const prev = document.createElement('button');
  prev.type = 'button'; prev.className = 'btn ghost'; prev.textContent = '← previous';
  const next = document.createElement('button');
  next.type = 'button'; next.className = 'btn ghost'; next.textContent = 'next →';
  prev.addEventListener('click', () => { state.stage = Math.max(0, state.stage - 1); input.value = String(state.stage); render(); });
  next.addEventListener('click', () => { state.stage = Math.min(STAGES.length - 1, state.stage + 1); input.value = String(state.stage); render(); });
  nav.append(prev, next);
  rail.append(ticks, input, nav);
  root.appendChild(rail);

  const panel = document.createElement('div');
  panel.className = 'explorable-body trace-panel';
  root.appendChild(panel);

  function render() {
    ticks.querySelectorAll('.scrub-tick').forEach((t, i) => t.classList.toggle('is-on', i === state.stage));
    const q = d.questions[state.q];
    panel.innerHTML = '';
    const stage = STAGES[state.stage];
    if (stage === 'state') {
      panel.appendChild(p(`"${d.state.text}"`, 't-mono trace-text'));
      panel.appendChild(p(`${d.state.n_tokens} tokens (+1 eos sink) → Ls = ${d.Ls}`));
    } else if (stage === 'trunk') {
      panel.appendChild(p(`${d.trunk.backbone}, layers 1–${d.trunk.tap_layer.split(' ')[0]} of ${d.trunk.tap_layer.split(' ')[2]}`));
      panel.appendChild(p(d.trunk.lora));
      panel.appendChild(p(`${d.trunk.h100_state_ms} ms (H100, one-off) — read once, before any question`, 'muted'));
    } else if (stage === 'cache') {
      panel.appendChild(p(`KV cache: ${d.Ls} positions, reused by every question below`));
    } else if (stage === 'suffix') {
      panel.appendChild(p(`"${q.suffix_text.trim()}"`, 't-mono trace-text'));
      panel.appendChild(p(`${q.suffix_tokens} tokens, T=${q.T} · render: ${q.render}${q.is_bern ? ' (Bernoulli — candidates not rendered)' : ''}`));
      if (q.type === 'noul') {
        const swap = d.noul_swap;
        panel.appendChild(p(`label-swap: [${swap.order_a.join(', ')}] vs [${swap.order_b.join(', ')}] → identical suffix: ${swap.suffix_identical}`, 'note'));
      }
    } else if (stage === 'head') {
      panel.appendChild(p(q.type === 'noul' ? d.head.noul_formula : d.head.choice_formula, 't-mono trace-text'));
    } else if (stage === 'probs') {
      const rows = Object.entries(q.probs).map(([label, pval]) => ({ label, p: pval }));
      rows.push({ label: '∅ none', p: q.p_null, isNull: true });
      const barsEl = document.createElement('div');
      panel.appendChild(barsEl);
      bars(barsEl, rows);
      if (q.expected != null) panel.appendChild(p(`E[index] = ${q.expected.toFixed(2)}`));
      panel.appendChild(p(`replayed from replays.json (${d.replay.model}, ${d.replay.device}, ${d.replay.ms} ms for all four questions)`, 'muted'));
    }
    sourceLine(panel, d.source);
  }

  function p(text, cls = '') {
    const n = document.createElement('p');
    if (cls) n.className = cls;
    n.textContent = text;
    return n;
  }

  render();
}

// ============================================================================
// P4 — run explorer (pivot by eval set / pivot by run; timeline folded in as x-axis)
// ============================================================================
async function mountRunExplorer() {
  const el = document.getElementById('chart-runs');
  if (!el) return;
  skeleton(el, 'data/runs-index.json');
  const [idx, timeline] = await Promise.all([loadJSON('data/runs-index.json'), loadJSON('data/research-timeline.json')]);
  if (!idx) return missing(el, 'data/runs-index.json');

  const runsWithDate = idx.runs.filter((r) => r.date);
  const GROUPS = [
    { value: 'lineage', label: 'lineage' },
    { value: 'ablation', label: 'ablations' },
    { value: 'probe', label: 'probes' },
    { value: 'baseline', label: 'baselines' }, // zero_shot folds into "baselines" here - one more toggle than the spec lists isn't worth it
  ];
  const activeGroups = new Set(['lineage']);
  const state = { pivot: 'set', set: idx.sets[0]?.key, metric: 'acc', run: runsWithDate[0]?.name };

  el.innerHTML = '';
  const root = document.createElement('div');
  root.className = 'explorable explorable-runs';
  el.appendChild(root);

  const pivotRow = document.createElement('div');
  pivotRow.className = 'explorable-row';
  pivotRow.appendChild(seg([{ value: 'set', label: 'pick an eval set' }, { value: 'run', label: 'pick a run' }], state.pivot, (v) => { state.pivot = v; render(); }, 'pivot'));
  root.appendChild(pivotRow);

  const controlsA = document.createElement('div');
  controlsA.className = 'explorable-row';
  const setSelect = document.createElement('select');
  setSelect.className = 'btn ghost';
  let curFam = null;
  let optgroup = null;
  idx.sets.forEach((s) => {
    if (s.family !== curFam) { curFam = s.family; optgroup = document.createElement('optgroup'); optgroup.label = curFam; setSelect.appendChild(optgroup); }
    const opt = document.createElement('option');
    opt.value = s.key; opt.textContent = s.key;
    (optgroup || setSelect).appendChild(opt);
  });
  setSelect.value = state.set;
  setSelect.addEventListener('change', () => { state.set = setSelect.value; render(); });
  controlsA.append(setSelect, seg([{ value: 'acc', label: 'acc' }, { value: 'nll', label: 'nll' }], state.metric, (v) => { state.metric = v; render(); }, 'metric'), checks(GROUPS, activeGroups, () => render(), 'show'));

  const controlsB = document.createElement('div');
  controlsB.className = 'explorable-row';
  const runSelect = document.createElement('select');
  runSelect.className = 'btn ghost';
  runsWithDate.forEach((r) => {
    const opt = document.createElement('option');
    opt.value = r.name; opt.textContent = `${r.name} (${r.date})`;
    runSelect.appendChild(opt);
  });
  runSelect.value = state.run;
  runSelect.addEventListener('change', () => { state.run = runSelect.value; render(); });
  controlsB.appendChild(runSelect);

  root.append(controlsA, controlsB);

  const body = document.createElement('div');
  root.appendChild(body);

  function toDay(iso) {
    const d0 = new Date('2026-09-16T00:00:00Z');
    return (new Date(iso + 'T00:00:00Z') - d0) / 86400000;
  }
  function fmtDay(day) {
    const dt = new Date(Date.UTC(2026, 8, 16) + day * 86400000);
    return `${String(dt.getUTCMonth() + 1).padStart(2, '0')}-${String(dt.getUTCDate()).padStart(2, '0')}`;
  }

  function renderSetPivot() {
    const rows = runsWithDate.filter((r) => activeGroups.has(r.group) && r.eval[state.set]);
    body.innerHTML = '';
    if (!rows.length) { body.appendChild(p('no runs with this set in the selected groups', 'note')); return; }
    const w = 1040, h = 360, mm = { t: 20, r: 100, b: 50, l: 50 };
    const iw = w - mm.l - mm.r, ih = h - mm.t - mm.b;
    const days = rows.map((r) => toDay(r.date));
    const maxDay = Math.max(...days, 5);
    const vals = rows.map((r) => r.eval[state.set][state.metric]).filter(isNum);
    const yMax = state.metric === 'nll' ? Math.max(...vals, 1) * 1.1 : 1;
    const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', role: 'img' });
    const container = document.createElement('div');
    container.style.background = '#fff';
    container.style.maxWidth = `${w}px`;
    container.appendChild(svg);
    const g = svgEl('g', { transform: `translate(${mm.l},${mm.t})` });
    svg.appendChild(g);
    const x = scale([0, maxDay], [0, iw]);
    const y = scale([0, yMax], [ih, 0]);
    niceTicks(0, yMax, 5).forEach((t) => {
      g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y(t), y2: y(t), stroke: STEEL, 'stroke-dasharray': '2,3' }));
      g.appendChild(text(-8, y(t) + 3, fmtNum(t), { fill: MID, 'text-anchor': 'end' }));
    });
    for (let day = 0; day <= Math.ceil(maxDay); day++) {
      g.appendChild(text(x(day), ih + 18, fmtDay(day), { fill: MID, 'text-anchor': 'middle' }));
    }
    const floor = idx.floors[state.set];
    if (floor && state.metric === 'acc') {
      g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: y(floor.value), y2: y(floor.value), stroke: MID, 'stroke-dasharray': '4,3' }));
      g.appendChild(text(iw - 4, y(floor.value) - 5, `floor ${floor.value.toFixed(3)}`, { fill: MID, 'text-anchor': 'end', 'font-size': 11 }));
    }
    rows.forEach((r) => {
      const val = r.eval[state.set][state.metric];
      if (!isNum(val)) return;
      const cx = x(toDay(r.date));
      const cy = y(val);
      const released = !!r.released_as;
      const dot = svgEl('circle', { cx, cy, r: released ? 6 : 5, fill: released ? INK : 'none', stroke: INK, 'stroke-width': released ? 0 : 1.6, ...(released ? {} : { 'stroke-dasharray': '2,1.5' }) });
      const ttl = svgEl('title');
      ttl.textContent = `${r.name} · ${r.date} · ${state.metric} ${val.toFixed(3)}${released ? ` · released as ${r.released_as}` : ' · (not released)'}`;
      dot.appendChild(ttl);
      g.appendChild(dot);
    });
    (timeline?.bugs || []).forEach((b) => {
      const bx = x(toDay(b.date));
      if (bx < 0 || bx > iw) return;
      g.appendChild(svgEl('line', { x1: bx, x2: bx, y1: 0, y2: ih, stroke: STEEL, 'stroke-dasharray': '1,3' }));
    });
    g.appendChild(svgEl('line', { x1: 0, x2: 0, y1: 0, y2: ih, stroke: STEEL }));
    g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: ih, y2: ih, stroke: STEEL }));
    body.appendChild(container);
    const cap = p(floor ? `floor: ${floor.value.toFixed(3)} (${floor.source})` : 'no sourced floor for this set', 'note');
    body.appendChild(cap);
    sourceLine(body, idx.source);
  }

  function renderRunPivot() {
    body.innerHTML = '';
    const run = idx.runs.find((r) => r.name === state.run);
    if (!run) return;
    const w = 1040, h = 220, mm = { t: 20, r: 20, b: 40, l: 20 };
    const iw = w - mm.l - mm.r, ih = h - mm.t - mm.b;
    const sets = idx.sets.filter((s) => run.eval[s.key]);
    const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', role: 'img' });
    const container = document.createElement('div');
    container.style.background = '#fff';
    container.style.maxWidth = `${w}px`;
    container.appendChild(svg);
    const g = svgEl('g', { transform: `translate(${mm.l},${mm.t})` });
    svg.appendChild(g);
    const x = scale([0, Math.max(sets.length - 1, 1)], [0, iw]);
    const y = scale([0, 1], [ih, 0]);
    g.appendChild(svgEl('line', { x1: 0, x2: iw, y1: ih, y2: ih, stroke: STEEL }));
    sets.forEach((s, i) => {
      const acc = run.eval[s.key].acc;
      const cx = x(i);
      const floor = idx.floors[s.key];
      if (floor) g.appendChild(svgEl('line', { x1: cx, x2: cx, y1: ih, y2: ih + 5, stroke: MID }));
      if (!isNum(acc)) return;
      const cy = y(acc);
      const dot = svgEl('circle', { cx, cy, r: 4, fill: INK });
      const ttl = svgEl('title');
      ttl.textContent = `${s.key} (${s.family}) · acc ${acc.toFixed(3)}${floor ? ` · floor ${floor.value.toFixed(3)}` : ''}`;
      dot.appendChild(ttl);
      g.appendChild(dot);
    });
    body.appendChild(container);
    body.appendChild(p(`${sets.length} sets · ${run.date || 'date not sourced'} · ${run.group}${run.released_as ? ` · released as ${run.released_as}` : ''}`, 'note'));
    sourceLine(body, idx.source);
  }

  function text(x, y, str, attrs = {}) {
    const t = svgEl('text', { x, y, 'font-size': 12, 'font-family': 'inherit', ...attrs });
    t.textContent = str;
    return t;
  }
  function p(str, cls = '') {
    const n = document.createElement('p');
    if (cls) n.className = cls;
    n.textContent = str;
    return n;
  }

  function render() {
    controlsA.style.display = state.pivot === 'set' ? '' : 'none';
    controlsB.style.display = state.pivot === 'run' ? '' : 'none';
    if (state.pivot === 'set') renderSetPivot(); else renderRunPivot();
  }
  render();
}

export function mountExplorables() {
  mountLatency();
  mountCalibration();
  mountTrace();
  mountRunExplorer();
}

if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', mountExplorables);
}
