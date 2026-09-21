// ink.js - monochrome probability primitives. Replaces heat.js entirely: no colour, ever,
// only ink density and serif/sans type. Three exports:
//   inkBars(el, {rows:[{label,p}], nullP})  - museum-label list, hairline track, ink fill
//   stipple(canvas, p, opts)                - engraving-style dot field, density ∝ p
//   ledger(el)                              - hero's decision register, push() to append
const REDUCED_MOTION = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;

export function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

// Pure - no DOM. rows:[{label,p}], nullP?number -> [{label,p,isNull}], ∅ row last.
export function normalizeRows(rows, nullP) {
  const out = (rows || []).map((r) => ({ label: String(r.label ?? ''), p: clamp01(r.p ?? 0), isNull: false }));
  if (nullP != null) out.push({ label: '∅', p: clamp01(nullP), isNull: true });
  return out;
}

export function mulberry32(seed) {
  let a = seed | 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- inkBars --------------------------------------------------------------------------
// inkBars(el, {rows, nullP}) -> {update(rows, nullP)}. Label in Inter 12px uppercase
// tracked (CSS handles the casing/tracking), a hairline track with an ink fill sized to
// p, value as a serif numeral right-aligned. The ∅ row (if any) renders last, outline
// only - no fill - via .ink-bar-null in CSS.
export function inkBars(el, { rows = [], nullP = null } = {}) {
  el.classList.add('ink-bars');
  let entries = [];

  function build(list) {
    el.innerHTML = '';
    entries = list.map((r) => {
      const row = document.createElement('div');
      row.className = 'ink-bar-row' + (r.isNull ? ' ink-bar-null' : '');
      const label = document.createElement('span');
      label.className = 'ink-bar-label';
      label.textContent = r.label;
      const track = document.createElement('span');
      track.className = 'ink-bar-track';
      const fill = document.createElement('span');
      fill.className = 'ink-bar-fill';
      track.appendChild(fill);
      const value = document.createElement('span');
      value.className = 'ink-bar-value';
      row.append(label, track, value);
      el.appendChild(row);
      return { fill, value, p: r.p };
    });
  }

  function paint(animate) {
    entries.forEach(({ fill, value, p }) => {
      value.textContent = p.toFixed(2);
      const pct = `${(p * 100).toFixed(1)}%`;
      if (animate && !REDUCED_MOTION && typeof gsap !== 'undefined') {
        gsap.to(fill, { width: pct, duration: 0.5, ease: 'power2.out' });
      } else {
        fill.style.width = pct;
      }
    });
  }

  function update(nextRows, nextNullP) {
    build(normalizeRows(nextRows, nextNullP));
    paint(true);
  }

  update(rows, nullP);
  return { update };
}

// ---- stipple ----------------------------------------------------------------------------
// stipple(canvas, p, opts) -> {update(p), destroy()}. Engraving-style stipple field: a grid
// of candidate dots, each kept with probability p (jittered off-grid so it doesn't read as
// a screen tone), ink on transparent. devicePixelRatio-aware. Deterministic per opts.seed so
// re-renders (resize) don't flicker to a different dot pattern for the same p.
export function stipple(canvas, p = 0, opts = {}) {
  const cell = opts.cell ?? 22;
  const seed = opts.seed ?? 1;
  let value = clamp01(p);

  function draw() {
    const ratio = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * ratio));
    const h = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = opts.color ?? '#000';
    const rng = mulberry32(seed); // reset each draw - same p -> same dots
    for (let y = cell / 2; y < rect.height; y += cell) {
      for (let x = cell / 2; x < rect.width; x += cell) {
        const keep = rng();
        if (keep > value) continue;
        const jx = (rng() - 0.5) * cell * 0.6;
        const jy = (rng() - 0.5) * cell * 0.6;
        const r = 0.6 + rng() * cell * 0.16 * value;
        ctx.beginPath();
        ctx.arc(x + jx, y + jy, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  draw();
  const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(draw) : null;
  if (ro) ro.observe(canvas);

  return {
    update(nextP) {
      value = clamp01(nextP);
      draw();
    },
    destroy() {
      if (ro) ro.disconnect();
    },
  };
}

// ---- ledger -------------------------------------------------------------------------
// ledger(el) -> {push(entry)}. entry: {text, decision, p, ms}. Rows are prepended (newest
// on top), revealed with a GSAP fade/rise, and the list is trimmed to the last 12 - this
// is a register, not an infinite log.
const LEDGER_MAX_ROWS = 12;

export function ledger(el) {
  el.classList.add('ink-ledger');
  let idx = 0;

  function push(entry) {
    idx += 1;
    const row = document.createElement('div');
    row.className = 'ledger-row';

    const n = document.createElement('span');
    n.className = 'ledger-index';
    n.textContent = String(idx).padStart(3, '0');

    const state = document.createElement('span');
    state.className = 'ledger-state';
    state.textContent = entry.text ?? '';

    const decision = document.createElement('span');
    decision.className = 'ledger-decision';
    decision.textContent = entry.decision ?? '';

    const p = document.createElement('span');
    p.className = 'ledger-p';
    p.textContent = entry.p != null ? entry.p.toFixed(2) : '—';

    const ms = document.createElement('span');
    ms.className = 'ledger-ms';
    ms.textContent = entry.ms != null ? `${entry.ms.toFixed(0)} ms` : '';

    row.append(n, state, decision, p, ms);
    el.insertBefore(row, el.firstChild);

    if (!REDUCED_MOTION && typeof gsap !== 'undefined') {
      gsap.fromTo(row, { opacity: 0, y: -6 }, { opacity: 1, y: 0, duration: 0.4, ease: 'power2.out' });
    }
    while (el.children.length > LEDGER_MAX_ROWS) el.removeChild(el.lastChild);
  }

  return { push };
}

// ---- self-test (pure helpers only - no DOM needed under `node js/ink.js`) -----------------
function selfTest() {
  console.assert(clamp01(-1) === 0 && clamp01(2) === 1 && clamp01(0.4) === 0.4, 'clamp01 clamps to [0,1]');

  const rows = normalizeRows([{ label: 'a', p: 0.7 }, { label: 'b', p: 1.4 }], 0.05);
  console.assert(rows.length === 3, 'normalizeRows appends the ∅ row');
  console.assert(rows[2].isNull === true && rows[2].label === '∅', 'the ∅ row is last and marked isNull');
  console.assert(rows[1].p === 1, 'normalizeRows clamps out-of-range p');

  const noNull = normalizeRows([{ label: 'a', p: 0.5 }], null);
  console.assert(noNull.length === 1, 'normalizeRows omits ∅ when nullP is null');

  const r1 = mulberry32(7);
  const r2 = mulberry32(7);
  console.assert(r1() === r2() && r1() === r2(), 'mulberry32 is deterministic for a given seed');

  console.log('ink.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
