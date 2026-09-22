// 04 SQL - `SELECT * FROM tickets WHERE typical(body, '<condition>')`. 129 rows, one Noul
// query each. Register: query line, ghost-link conditions, each row a bars() readout
// (js/bars.js, one candidate wide) - is-winner once it's a match. The scan animates only
// by the rows filling in as they're decided - no glow.
//
// TEMPLATE (mirrored verbatim in scripts/record_data_demos.py's sql_question()):
//   state = row text, verbatim
//   query = {type:'noul', question:`Does this row satisfy the condition: ${condition}? Answer
//            yes if ${criterion}; otherwise answer no.`, labels:['no','yes']}
import { bars } from '../bars.js';

// One ticket's row: the full text as the bar's label (compact bars in exhibits.css give it the
// room), p as the fill. is-winner is our own >0.5 predicate, not bars.js's "max of the row" (a
// single-row bars() call is trivially its own max) - set by hand after each update.
function scanRow(text) {
  const el = document.createElement('div');
  el.title = text;
  const h = bars(el, [{ label: text, p: 0 }]);
  return {
    el,
    update(p, { isWinner = p != null && p >= 0.5 } = {}) {
      h.update([{ label: text, p: p ?? 0 }]);
      el.querySelector('.bar')?.classList.toggle('is-winner', isWinner);
    },
  };
}

function sqlQuestion(condition, criterion) {
  return `Does this row satisfy the condition: ${condition}? Answer yes if ${criterion}; otherwise answer no.`;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

const CHIP_LABEL = { D: 'duplicate charge', C: 'cancelling', B: 'arrived damaged', L: 'login / password' };
const VISIBLE_ROWS = 12;

export async function mount(el, ctx) {
  const res = await fetch('data/demos/sql.json');
  const pool = res.ok ? await res.json() : null;
  if (!pool) {
    el.innerHTML = '<p class="muted">sql pool unavailable.</p>';
    return;
  }

  el.innerHTML = '';
  el.className = 'exhibit sql-exhibit';

  const strip = document.createElement('p');
  strip.className = 'sql-query';
  el.appendChild(strip);

  const chips = document.createElement('div');
  chips.className = 'ex-chip-row';
  pool.conditions.forEach((cond) => {
    const chip = document.createElement('button');
    chip.className = 'ex-ghost';
    chip.type = 'button';
    chip.textContent = CHIP_LABEL[cond.id] ?? cond.id;
    chip.setAttribute('aria-pressed', 'false');
    chip.addEventListener('click', () => run(cond));
    chips.appendChild(chip);
    cond._chip = chip;
  });
  el.appendChild(chips);

  const readoutLine = document.createElement('p');
  readoutLine.className = 't-mono muted';
  readoutLine.textContent = 'Pick a condition to scan the table.';
  el.appendChild(readoutLine);

  const register = document.createElement('div');
  register.className = 'sql-register';
  el.appendChild(register);

  const showMore = document.createElement('button');
  showMore.type = 'button';
  showMore.className = 'ex-ghost';
  showMore.textContent = `Show all ${pool.rows.length} →`;
  el.appendChild(showMore);

  const caption = document.createElement('p');
  caption.className = 'ex-caption';
  caption.textContent =
    "129 rows, one forward pass each, no embeddings. Four conditions score precision 1.00 and recall .69–1.00 against hand labels. It keys on the words in the rule: 'charged once' lit up 'duplicate charge'.";
  el.appendChild(caption);

  const rowEls = pool.rows.map((row) => scanRow(row.text));

  // 12 rows shown by default (no scroll box); "Show all N →" renders the rest inline.
  let order = pool.rows.map((_, i) => i);
  let expanded = false;
  function renderVisible() {
    register.innerHTML = '';
    (expanded ? order : order.slice(0, VISIBLE_ROWS)).forEach((i) => register.appendChild(rowEls[i].el));
    showMore.hidden = expanded || pool.rows.length <= VISIBLE_ROWS;
  }
  showMore.addEventListener('click', () => {
    expanded = true;
    renderVisible();
  });
  renderVisible();

  let running = false;

  async function run(cond) {
    if (running) return;
    running = true;
    pool.conditions.forEach((c) => c._chip.setAttribute('aria-pressed', String(c === cond)));
    strip.innerHTML = `SELECT * FROM tickets WHERE typical(body, <span class="cond">'${cond.condition}'</span>);`;

    const question = sqlQuestion(cond.condition, cond.criterion);

    let tp = 0,
      fp = 0,
      fn = 0,
      scanned = 0;
    const t0 = performance.now();

    for (let i = 0; i < pool.rows.length; i++) {
      const row = pool.rows[i];
      const out = await ctx.decide(row.text, [{ type: 'noul', question, labels: ['no', 'yes'] }]);
      const p = out?.results?.[0]?.probs?.yes ?? 0;
      row._p = p;
      const predicted = p > 0.5;
      const gold = row.gold === cond.id;
      rowEls[i].update(p, { isWinner: predicted });
      if (predicted && gold) tp++;
      else if (predicted && !gold) fp++;
      else if (!predicted && gold) fn++;
      scanned++;
      const prec = tp + fp ? tp / (tp + fp) : 1;
      const rec = tp + fn ? tp / (tp + fn) : 1;
      readoutLine.textContent = `${scanned}/${pool.rows.length} rows · ${tp} matches · P ${prec.toFixed(2)} R ${rec.toFixed(2)}`;
      if (ctx.mode() !== 'live') await sleep(Math.min(out?.ms ?? 15, 35));
    }

    const wallMs = performance.now() - t0;
    const prec = tp + fp ? tp / (tp + fp) : 1;
    const rec = tp + fn ? tp / (tp + fn) : 1;
    readoutLine.textContent = `${pool.rows.length} rows · ${tp} matches · P ${prec.toFixed(2)} R ${rec.toFixed(2)} · ${(wallMs / 1000).toFixed(1)} s here`;

    // matches rise to the top
    order = pool.rows.map((_, i) => i).sort((a, b) => (pool.rows[b]._p ?? 0) - (pool.rows[a]._p ?? 0));
    renderVisible();

    running = false;
  }

  // default: run the first condition once the screen is visible
  run(pool.conditions[0]);
}
