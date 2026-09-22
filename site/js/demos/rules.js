// Rules are the program: same 5 facts, drag/nudge the rule list into a different
// precedence order, and the verdict re-decides live. The rubric itself (state, rules,
// question prefix, the two recorded orders) lives in data/presets.json's "flip" - already
// loaded by shell.js into ctx.presets, read from there rather than duplicated here.
// data/demos/rules.json only adds the 6 hand-written applicants for the second panel.
const OUTCOME_ORDER = ['deny', 'approve', 'approve_with_conditions', 'refer_to_underwriter'];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

// query_text's choice-branch rendering, ported (see inference/typical/core.py::query_text) -
// must match byte-for-byte or the replay hashKey won't hit the two recorded orders.
function orderQuery(flip, order) {
  const rubric = order.map((label) => `${label}: ${flip.rules[label]}`).join('  ');
  return `${flip.question_prefix}\n${rubric}`;
}

function sameOrder(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

export async function mount(host, ctx) {
  const [flip, rulesDoc] = await Promise.all([
    Promise.resolve(ctx.presets?.flip),
    fetch('data/demos/rules.json').then((res) => (res.ok ? res.json() : null)).catch(() => null),
  ]);
  if (!flip || !rulesDoc) {
    host.textContent = 'rules offline: data/presets.json or data/demos/rules.json missing';
    return;
  }

  host.innerHTML = '';
  const wrap = el('div', 'exhibit rules-exhibit');

  // ---- panel 1: facts, reorderable rules, live outcome ----
  // Two columns above 480px, one below - a CSS media query (exhibits.css), not a one-time width
  // measurement here: the old inline-style version measured host width once at mount and never
  // updated on resize, so a card mounted wide and then resized narrow (e.g. expanded at a wide
  // viewport, then the window/device shrinks) kept a stale 2-column grid and overflowed.
  const panel1 = el('div', 'rules-panel1');

  const left = el('div');
  left.appendChild(el('p', 'ex-subhead', 'The facts (fixed)'));
  const factsList = el('ol', 'rules-facts');
  flip.state.split(/(?<=\.)\s+/).forEach((sentence) => factsList.appendChild(el('li', null, sentence)));
  left.appendChild(factsList);

  left.appendChild(el('p', 'ex-subhead', 'The rules (reorder them)'));
  const ruleList = el('div', 'rules-order');
  left.appendChild(ruleList);
  const staticNote = el('p', 'ex-caption');
  left.appendChild(staticNote);

  const right = el('div');
  const bars = el('div', 'rules-live-bars');
  right.appendChild(bars);
  const readoutLine = el('p', 'ex-readout');
  right.appendChild(readoutLine);
  const field = ctx.inkBars(bars, { rows: [] });

  panel1.append(left, right);
  wrap.appendChild(panel1);

  let order = flip.orders[0].slice();

  function renderRuleList() {
    ruleList.innerHTML = '';
    order.forEach((label, i) => {
      const row = el('div', 'rules-order-row');
      const text = el('span', 'rules-order-text', `${i + 1}. ${label}: ${flip.rules[label]}`);
      const btns = el('span', 'rules-order-btns');
      const up = el('button', 'ex-ghost', '↑');
      const down = el('button', 'ex-ghost', '↓');
      up.type = 'button';
      down.type = 'button';
      up.disabled = i === 0;
      down.disabled = i === order.length - 1;
      up.addEventListener('click', () => move(i, i - 1));
      down.addEventListener('click', () => move(i, i + 1));
      btns.append(up, down);
      row.append(text, btns);
      ruleList.appendChild(row);
    });
  }

  async function move(from, to) {
    if (to < 0 || to >= order.length) return;
    const next = order.slice();
    [next[from], next[to]] = [next[to], next[from]];
    if (ctx.mode() !== 'live' && !flip.orders.some((o) => sameOrder(o, next))) {
      staticNote.textContent = 'Static mode: only the two recorded orders (deny-first / approve-first) are available. Run the live server to try others.';
      return;
    }
    staticNote.textContent = '';
    order = next;
    renderRuleList();
    await redecide();
  }

  async function redecide() {
    const question = orderQuery(flip, order);
    const t0 = performance.now();
    const res = await ctx.decide(flip.state, [{ type: 'choice', question, labels: order }]);
    const ms = res?.ms ?? performance.now() - t0;
    if (!res) {
      readoutLine.textContent = 'No recorded result for this order (offline).';
      return;
    }
    const r = res.results[0];
    const rows = OUTCOME_ORDER.map((label) => ({ label, p: r.probs[label] ?? 0 }));
    field.update(rows, r.p_null ?? null);
    readoutLine.textContent = `${order[0]} checked first → ${r.argmax} (${r.probs[r.argmax].toFixed(2)}) · ${ms.toFixed(0)} ms`;
  }

  renderRuleList();
  await redecide();

  // ---- panel 2: 6 hand-written applicants x 2 orders, one decide() call each ----
  const panel2 = el('div', 'rules-panel2');
  panel2.style.marginTop = '8px';
  panel2.appendChild(el('p', 'ex-subhead', 'Same rubric, 6 applicants'));
  const tableWrap = el('div', 'table-scroll');
  const table = document.createElement('table');
  table.className = 'ex-table';
  const thead = document.createElement('thead');
  thead.innerHTML = '<tr><th>applicant</th><th>deny checked first</th><th>approve checked first</th></tr>';
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  table.appendChild(tbody);
  tableWrap.appendChild(table);
  panel2.appendChild(tableWrap);
  const panel2Caption = el('p', 'ex-caption', rulesDoc.caption);
  panel2.appendChild(panel2Caption);
  wrap.appendChild(panel2);
  host.appendChild(wrap);

  const orderQueries = flip.orders.map((o) => ({ type: 'choice', question: orderQuery(flip, o), labels: o }));
  for (const applicant of rulesDoc.applicants) {
    const tr = document.createElement('tr');
    const nameTd = document.createElement('td');
    nameTd.className = 'rules-applicant';
    nameTd.textContent = applicant.label;
    tr.appendChild(nameTd);
    const res = await ctx.decide(applicant.state, orderQueries);
    if (!res) {
      tr.append(el('td', 'rules-applicant', '—'), el('td', 'rules-applicant', '—'));
    } else {
      const [denyFirst, approveFirst] = res.results;
      const flipped = denyFirst.argmax !== approveFirst.argmax;
      [denyFirst, approveFirst].forEach((r, i) => {
        const td = document.createElement('td');
        const cell = el('div', 'ex-cell ex-cell--col');
        const word = el('span', 'ex-cell-word', r.argmax);
        word.title = r.argmax;
        const barRow = el('span', 'ex-cell-barrow');
        const strip = el('span', 'dotbars-strip');
        const dots = 10;
        const lit = Math.round(r.probs[r.argmax] * dots);
        for (let d = 0; d < dots; d++) strip.appendChild(el('i', d < lit ? 'on' : null));
        const num = el('span', 'ex-num', r.probs[r.argmax].toFixed(2));
        barRow.append(strip, num);
        cell.append(word, barRow);
        if (i === 1 && flipped) cell.appendChild(el('span', 'ex-tag ex-tag--flip', 'FLIP'));
        td.appendChild(cell);
        tr.appendChild(td);
      });
    }
    tbody.appendChild(tr);
  }
}
