// 04 SQL - `SELECT * FROM tickets WHERE typical(body, '<condition>')`. 129 rows, one Noul
// query each. Presentation is a museum register: chalk specimen strip, ghost-text
// condition chips, hairline rows with a 2px ink bar (width ∝ p) + serif numeral - no
// colour, ever (see js/ink.js / exhibits.css).
//
// TEMPLATE (mirrored verbatim in scripts/record_data_demos.py's sql_question()):
//   state = row text, verbatim
//   query = {type:'noul', question:`Does this row satisfy the condition: ${condition}? Answer
//            yes if ${criterion}; otherwise answer no.`, labels:['no','yes']}

function sqlQuestion(condition, criterion) {
  return `Does this row satisfy the condition: ${condition}? Answer yes if ${criterion}; otherwise answer no.`;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function fmtRow(text, n = 92) {
  return text.length > n ? text.slice(0, n) + '…' : text;
}

const CHIP_LABEL = { D: 'duplicate charge', C: 'cancelling', B: 'arrived damaged', L: 'login / password' };

export async function mount(el, ctx) {
  const res = await fetch('data/demos/sql.json');
  const pool = res.ok ? await res.json() : null;
  if (!pool) {
    el.innerHTML = '<p class="dim">sql pool unavailable.</p>';
    return;
  }
  const section = el.closest('.screen');
  const readoutEl = section?.querySelector('.readout');

  el.innerHTML = '';
  el.className = 'exhibit sql-exhibit';

  const strip = document.createElement('div');
  strip.className = 'chalk-strip';
  el.appendChild(strip);

  const chips = document.createElement('div');
  chips.className = 'chip-row';
  pool.conditions.forEach((cond) => {
    const chip = document.createElement('button');
    chip.className = 'chip-ghost';
    chip.type = 'button';
    chip.textContent = CHIP_LABEL[cond.id] ?? cond.id;
    chip.setAttribute('aria-pressed', 'false');
    chip.addEventListener('click', () => run(cond));
    chips.appendChild(chip);
    cond._chip = chip;
  });
  el.appendChild(chips);

  const readoutLine = document.createElement('p');
  readoutLine.className = 'exhibit-readout';
  readoutLine.textContent = 'pick a condition to scan the table.';
  el.appendChild(readoutLine);

  const register = document.createElement('div');
  register.className = 'register sql-register';
  el.appendChild(register);

  const caption = document.createElement('p');
  caption.className = 'exhibit-caption';
  caption.textContent =
    "129 rows, one forward pass each, no embeddings: 4 conditions at precision 1.00 and recall .69–1.00 against hand labels; it keys on the words in the rule — 'charged once' lit up 'duplicate charge'.";
  el.appendChild(caption);

  const rowEls = pool.rows.map((row) => {
    const div = document.createElement('div');
    div.className = 'reg-row';
    const text = document.createElement('span');
    text.className = 'reg-text';
    text.textContent = fmtRow(row.text);
    const bar = document.createElement('span');
    bar.className = 'reg-bar';
    const fill = document.createElement('span');
    fill.className = 'reg-bar-fill';
    bar.appendChild(fill);
    const p = document.createElement('span');
    p.className = 'reg-numeral';
    p.textContent = '—';
    div.append(text, bar, p);
    register.appendChild(div);
    return { div, fill, p };
  });

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
    let lastMs = null,
      lastDevice = null;
    const t0 = performance.now();

    for (let i = 0; i < pool.rows.length; i++) {
      const row = pool.rows[i];
      const { div, fill, p: pEl } = rowEls[i];
      rowEls.forEach((r) => r.div.classList.remove('scanning'));
      div.classList.add('scanning');
      const out = await ctx.decide(row.text, [{ type: 'noul', question, labels: ['no', 'yes'] }]);
      const p = out?.results?.[0]?.probs?.yes ?? 0;
      lastMs = out?.ms ?? lastMs;
      lastDevice = out?.device ?? lastDevice;
      row._p = p;
      fill.style.width = `${(p * 100).toFixed(1)}%`;
      pEl.textContent = p.toFixed(2);
      const predicted = p > 0.5;
      const gold = row.gold === cond.id;
      if (predicted && gold) tp++;
      else if (predicted && !gold) fp++;
      else if (!predicted && gold) fn++;
      scanned++;
      const prec = tp + fp ? tp / (tp + fp) : 1;
      const rec = tp + fn ? tp / (tp + fn) : 1;
      readoutLine.textContent = `${scanned}/${pool.rows.length} rows · ${tp} matches · P ${prec.toFixed(2)} R ${rec.toFixed(2)}`;
      if (ctx.mode() !== 'live') await sleep(Math.min(lastMs ?? 15, 35));
    }
    rowEls.forEach((r) => r.div.classList.remove('scanning'));

    const wallMs = performance.now() - t0;
    const prec = tp + fp ? tp / (tp + fp) : 1;
    const rec = tp + fn ? tp / (tp + fn) : 1;
    readoutLine.textContent = `${pool.rows.length} rows · ${tp} matches · P ${prec.toFixed(2)} R ${rec.toFixed(2)} · ${(wallMs / 1000).toFixed(1)} s here`;
    ctx.readout(readoutEl, { ms: lastMs, device: lastDevice, extra: `${tp}/${pool.rows.length} matched · P ${prec.toFixed(2)} R ${rec.toFixed(2)}` });

    // matches rise to the top
    rowEls
      .map((r, i) => ({ r, p: pool.rows[i]._p }))
      .sort((a, b) => b.p - a.p)
      .forEach(({ r }) => register.appendChild(r.div));

    running = false;
  }

  // default: run the first condition once the screen is visible
  run(pool.conditions[0]);
}
