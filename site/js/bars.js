// bars.js — the one probability readout: label + violet bar + value, ∅ as a dashed outline row.
// Used by the hero card, every demo, Try-it and the game chrome. DOM-only, no canvas.
//
//   bars(el, rows, {compact})  rows: [{label, p, isNull}]  -> { update(rows) }
//   inkBars(el, {rows, nullP}) legacy adapter for the demo modules -> { update(rows, nullP) }

function row(r, compact) {
  const el = document.createElement('div');
  el.className = 'bar' + (r.isNull ? ' is-null' : '') + (compact ? ' is-compact' : '');
  const label = document.createElement('span');
  label.className = 'bar-label';
  label.textContent = r.label; // may be user-typed (Try it): text only
  label.title = r.label;
  const track = document.createElement('div');
  track.className = 'bar-track';
  const fill = document.createElement('div');
  fill.className = 'bar-fill';
  const pct = Math.max(0, Math.min(1, r.p));
  fill.style.width = (pct * 100).toFixed(1) + '%';
  const val = document.createElement('span');
  val.className = 'bar-val' + (pct < 0.15 ? ' is-outside' : '');
  val.textContent = pct.toFixed(2);
  if (pct < 0.15) val.style.left = `calc(${(pct * 100).toFixed(1)}% + 8px)`;
  track.append(fill, val);
  el.append(label, track);
  return el;
}

export function bars(el, rows, { compact = false } = {}) {
  el.classList.add('bars');
  const render = (rs) => {
    el.innerHTML = '';
    const max = Math.max(...rs.filter((r) => !r.isNull).map((r) => r.p), -1);
    rs.forEach((r) => {
      const node = row(r, compact);
      if (!r.isNull && r.p === max) node.classList.add('is-winner');
      el.appendChild(node);
    });
  };
  render(rows);
  return { update: render };
}

export function inkBars(el, { rows = [], nullP = null } = {}) {
  const toRows = (rs, np) => [
    ...rs.map((r) => ({ label: r.label, p: r.p })),
    ...(np != null ? [{ label: '∅', p: np, isNull: true }] : []),
  ];
  const h = bars(el, toRows(rows, nullP));
  return { update: (rs, np) => h.update(toRows(rs, np)) };
}

// Turn an /api/decide result into rows for one query.
export function rowsFromResult(result) {
  return Object.entries(result.probs || {})
    .map(([label, p]) => ({ label, p }))
    .concat([{ label: '∅', p: result.p_null ?? 0, isNull: true }]);
}
