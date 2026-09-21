// One document, 20 questions (H2): the handbook is cached once, then all 20 typed
// questions (12 Noul, 5 Choice, 3 Score) are answered in a single ctx.decide() call. A
// second call asks only the first question, to show the one-at-a-time cost for comparison.
// ponytail: skipped hover-highlight-the-relevant-sentence (spec allows skipping it "if
// cheap, else skip" - precise sentence attribution per question isn't cheap here); add a
// per-question sentence-index field to data/demos/doc20.json if that lands later.

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function isHit(question, result) {
  return result.argmax === question.gold;
}

export async function mount(host, ctx) {
  const data = await fetch('data/demos/doc20.json')
    .then((res) => (res.ok ? res.json() : null))
    .catch(() => null);
  if (!data) {
    host.textContent = 'doc-20 offline: data/demos/doc20.json missing';
    return;
  }

  host.innerHTML = '';
  const wrap = el('div', 'exhibit doc20-exhibit');

  const handbook = document.createElement('details');
  handbook.className = 'ex-details';
  const summary = el('summary', null, 'Read the handbook');
  const pre = el('pre', null, data.doc);
  handbook.append(summary, pre);
  wrap.appendChild(handbook);

  const register = el('div', 'doc20-register');
  wrap.appendChild(register);

  const timing = el('div', 'ex-stat-row');
  const onePass = el('div', 'ex-stat');
  const oneAt = el('div', 'ex-stat');
  timing.append(onePass, oneAt);
  wrap.appendChild(timing);

  const caption = el('p', 'ex-caption');
  wrap.appendChild(caption);

  host.appendChild(wrap);

  const rows = data.questions.map((question) => {
    const row = el('div', 'ex-row');
    const q = el('span', 'doc20-q ex-row-text', question.display);
    const a = el('span', 'doc20-a', '');
    const mark = el('span', 'doc20-mark', '·');
    const p = el('span', 'ex-num', '');
    row.append(q, a, mark, p);
    register.appendChild(row);
    return { row, mark, a, p };
  });

  const allQueries = data.questions.map((question) => ({ type: question.type, question: question.query, labels: question.labels }));

  // single-question cost measured first, then the full 20-in-one-pass call.
  const single = await ctx.decide(data.doc, allQueries.slice(0, 1));
  const full = await ctx.decide(data.doc, allQueries);
  if (!full) {
    caption.textContent = 'No recorded result for this handbook (offline).';
    return;
  }

  let correct = 0;
  full.results.forEach((result, i) => {
    const question = data.questions[i];
    const t = rows[i];
    const hit = isHit(question, result);
    if (hit) correct += 1;
    const p = result.probs[result.argmax] ?? 0;
    t.mark.className = 'doc20-mark ' + (hit ? 'hit' : 'miss');
    t.a.textContent = result.argmax;
    t.p.textContent = p.toFixed(2);
  });

  function statPair(target, seconds, label) {
    target.innerHTML = '';
    target.append(el('span', 'ex-stat-num', `${seconds.toFixed(1)} s`), el('span', 'ex-stat-label', label));
  }
  statPair(onePass, full.ms / 1000, 'one pass, here');
  if (single) statPair(oneAt, (single.ms * data.questions.length) / 1000, 'one at a time');

  caption.textContent = `${correct}/20 here. Lookups and yes/no answers mostly land; ordered levels mostly do not. Every miss is a number compared against a threshold in the text.`;
}
