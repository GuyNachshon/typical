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

// Two rows are the same row if they carry the same label and the same ∅ status. Probability is
// deliberately not part of this: a changed probability is exactly the case we want to animate.
export function shapeOf(rows) {
  return rows.map((r) => `${r.label}\u0000${r.isNull ? 1 : 0}`).join('\u0001');
}

export function bars(el, rows) {
  el.classList.add('bars');
  // A node created with its width already set has no start value, so rebuilding every row on every
  // tick meant `.bar .fill`'s `transition: width` (src.css) never once fired — the bars have always
  // snapped. When the row shape is unchanged, update in place and let the browser tween. The label
  // set genuinely does change between ticks (snake's candidates are whichever moves are legal), so
  // the rebuild path stays for that case.
  let shape = null;
  const render = (rs) => {
    const max = Math.max(-1, ...rs.filter((r) => !r.isNull).map((r) => r.p));
    const next = shapeOf(rs);
    if (next === shape && el.children.length === rs.length) {
      rs.forEach((r, i) => {
        const node = el.children[i];
        const pct = Math.max(0, Math.min(1, r.p));
        node.querySelector('.fill').style.width = `${(pct * 100).toFixed(1)}%`;
        node.querySelector('.v').textContent = pct.toFixed(2);
        // the winner moves between ticks; a stale bold label is worse than the snap this replaces
        node.classList.toggle('is-winner', !r.isNull && r.p === max);
      });
      return;
    }
    shape = next;
    el.innerHTML = '';
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

export function selfTest() {
  const a = [{ label: 'up', p: 0.1 }, { label: 'down', p: 0.9 }, { label: '∅ none', p: 0.01, isNull: true }];
  console.assert(shapeOf(a) === shapeOf(a.map((r) => ({ ...r, p: Math.random() }))), 'a moved probability keeps the shape, so the bar animates');
  console.assert(shapeOf(a) !== shapeOf(a.slice(0, 2)), 'a dropped candidate changes it, so the rows rebuild');
  console.assert(shapeOf(a) !== shapeOf([{ label: 'up', p: 0.1, isNull: true }, ...a.slice(1)]), 'and so does a row becoming ∅');
  console.assert(shapeOf([{ label: 'a\u0001b', p: 0 }]) !== shapeOf([{ label: 'a', p: 0 }, { label: 'b', p: 0 }]), 'a label containing the separator cannot forge a different shape');
  console.assert(rowsFromResult({ probs: { x: 0.5 }, p_null: 0.5 }).length === 2, 'a result yields its labels plus ∅');
  console.log('bars.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}
