// rowdots.js — shared per-row dot-strip readout for exhibits whose probabilities are
// independent per row (SQL's 129-row scan, DOC-20's 20 answers, ADS's three decisions,
// COMPACTION's ranking): each row's champagne-ness is its own threshold, not "highest of
// this batch" (that's js/dots.js's dotBars, built for a single choice among candidates).
// Reuses the .dotbars-row/.dotbars-strip/.dotbars-label/.dotbars-num CSS classes (style.css)
// so it reads as the same primitive as every other probability on the page.
export function dotRow(label, { dots = 16, title } = {}) {
  const row = document.createElement('div');
  row.className = 'dotbars-row';
  const lbl = document.createElement('span');
  lbl.className = 'dotbars-label';
  lbl.textContent = label;
  if (title) lbl.title = title;
  const strip = document.createElement('span');
  strip.className = 'dotbars-strip';
  const cells = Array.from({ length: dots }, () => {
    const d = document.createElement('i');
    strip.appendChild(d);
    return d;
  });
  const num = document.createElement('span');
  num.className = 'dotbars-num';
  num.textContent = '—';
  row.append(lbl, strip, num);

  function update(p, { isWinner = p != null && p >= 0.5, isNull = false } = {}) {
    const lit = p == null ? 0 : Math.round(Math.max(0, Math.min(1, p)) * dots);
    cells.forEach((c, i) => { c.className = i < lit ? 'on' : ''; });
    row.className = 'dotbars-row' + (isNull ? ' is-null' : '') + (isWinner ? ' is-winner' : '');
    num.textContent = p == null ? '—' : p.toFixed(2);
  }
  return { el: row, update };
}

function selfTest() {
  const r = dotRow('test', { dots: 10 });
  console.assert(r.el.className === 'dotbars-row', 'dotRow starts unlit, not a winner');
  r.update(0.73, { isWinner: true });
  const lit = r.el.querySelectorAll('.dotbars-strip i.on').length;
  console.assert(lit === 7, `dotRow lights round(0.73*10)=7 dots, got ${lit}`);
  console.assert(r.el.classList.contains('is-winner'), 'dotRow.update applies is-winner');
  console.log('rowdots.js self-test OK');
  return true;
}

// Needs a DOM (dotRow builds real elements) — runs under a browser or jsdom, skipped under
// plain node (there's no document to build into).
if (typeof document !== 'undefined' && typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
