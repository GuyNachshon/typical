// bars.js — the one readout: label · ink bar on a light-gray track · value. ∅ last, dashed.
//
//   bars(el, rows)                      rows: [{label, p, isNull}]  -> { update(rows) }
//   inkBars(el, {rows, nullP})          legacy adapter for the demo modules
//   rowsFromResult(result)              /api/decide result -> rows (∅ appended)

function row(r) {
  const el = document.createElement('div');
  el.className = 'bar' + (r.isNull ? ' is-null' : '');
  const label = document.createElement('span');
  label.className = 'label';
  label.textContent = r.label; // may be user-typed (Try it): text only
  label.title = r.label;
  const track = document.createElement('div');
  track.className = 'track';
  const fill = document.createElement('div');
  fill.className = 'fill';
  const pct = Math.max(0, Math.min(1, r.p));
  fill.style.width = (pct * 100).toFixed(1) + '%';
  track.appendChild(fill);
  const val = document.createElement('span');
  val.className = 'v';
  val.textContent = pct.toFixed(2);
  el.append(label, track, val);
  return el;
}

export function bars(el, rows) {
  el.classList.add('bars');
  const render = (rs) => {
    el.innerHTML = '';
    const max = Math.max(-1, ...rs.filter((r) => !r.isNull).map((r) => r.p));
    rs.forEach((r) => {
      const node = row(r);
      if (!r.isNull && r.p === max) node.classList.add('is-winner');
      el.appendChild(node);
    });
  };
  render(rows);
  return { update: render };
}

export function inkBars(el, { rows = [], nullP = null } = {}) {
  const to = (rs, np) => [...rs.map((r) => ({ label: r.label, p: r.p })), ...(np != null ? [{ label: '∅ none', p: np, isNull: true }] : [])];
  const h = bars(el, to(rows, nullP));
  return { update: (rs, np) => h.update(to(rs, np)) };
}

export function rowsFromResult(result) {
  return Object.entries(result.probs || {}).map(([label, p]) => ({ label, p })).concat([{ label: '∅ none', p: result.p_null ?? 0, isNull: true }]);
}
