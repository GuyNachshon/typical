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

const SVG_NS = 'http://www.w3.org/2000/svg';

function hexEl(filled) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'hex reg-hex' + (filled ? ' filled' : ''));
  svg.setAttribute('viewBox', '0 0 12 12');
  const poly = document.createElementNS(SVG_NS, 'polygon');
  poly.setAttribute('points', '6,0.5 11,3.25 11,8.75 6,11.5 1,8.75 1,3.25');
  svg.appendChild(poly);
  return svg;
}

function isHit(question, result) {
  return result.argmax === question.gold;
}

export async function mount(host, ctx) {
  const data = await fetch('data/demos/doc20.json')
    .then((res) => (res.ok ? res.json() : null))
    .catch(() => null);
  if (!data) {
    host.textContent = 'doc-20 offline — data/demos/doc20.json missing';
    return;
  }

  host.innerHTML = '';
  const wrap = el('div', 'exhibit doc20-exhibit');

  const handbook = el('div', 'doc20-handbook', data.doc);
  wrap.appendChild(handbook);

  const register = el('div', 'register doc20-register');
  wrap.appendChild(register);

  const timing = el('div', 'doc20-timing');
  const onePass = el('div', 'timing-pair');
  const oneAt = el('div', 'timing-pair');
  timing.append(onePass, oneAt);
  wrap.appendChild(timing);

  const caption = el('p', 'exhibit-caption');
  wrap.appendChild(caption);

  host.appendChild(wrap);

  const rows = data.questions.map((question) => {
    const row = el('div', 'reg-row');
    row.appendChild(el('p', 'doc20-q', question.display));
    const aLine = el('div', 'doc20-a-line');
    const a = el('span', 'doc20-a', '');
    const p = el('span', 'doc20-p', '');
    aLine.append(a, p);
    row.appendChild(aLine);
    register.appendChild(row);
    return { row, a, p };
  });

  const allQueries = data.questions.map((question) => ({ type: question.type, question: question.query, labels: question.labels }));

  // single-question cost measured first, then the full 20-in-one-pass call.
  const single = await ctx.decide(data.doc, allQueries.slice(0, 1));
  const full = await ctx.decide(data.doc, allQueries);
  if (!full) {
    caption.textContent = 'no recorded result for this handbook — offline.';
    return;
  }

  let correct = 0;
  full.results.forEach((result, i) => {
    const question = data.questions[i];
    const t = rows[i];
    const hit = isHit(question, result);
    if (hit) correct += 1;
    const p = result.probs[result.argmax] ?? 0;
    t.a.before(hexEl(hit));
    t.a.textContent = result.argmax;
    t.p.textContent = p.toFixed(2);
  });

  function timingPair(target, seconds, label) {
    target.innerHTML = '';
    target.append(el('span', 'timing-numeral serif', `${seconds.toFixed(1)} s`), el('span', 'timing-label', label));
  }
  timingPair(onePass, full.ms / 1000, 'one pass, here');
  if (single) timingPair(oneAt, (single.ms * data.questions.length) / 1000, 'one at a time');

  caption.textContent = `${correct}/20 here: lookups and yes/no answers mostly land, ordered levels mostly don't — every miss is a number compared against a threshold in the text.`;

  const readoutEl = host.closest('.screen')?.querySelector('.readout');
  if (readoutEl) ctx.readout(readoutEl, { ms: full.ms, extra: `${correct}/20` });
}
