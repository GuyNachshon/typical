// 00 TRY IT — free-form state/question/options, wired through the same ctx.decide() every
// other exhibit uses. Prefilled with the loan-rubric example (data/presets.json's "flip",
// first rule order) because that exact query string is recorded in data/replays.json
// (js/demos/rules.js sends the byte-identical question for orders[0]) — so the prefill
// answers correctly even in static mode. Any other input only resolves once the local
// server is live (ctx.decide() falls back to a replay lookup that won't exist for new text).

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

const TYPES = ['choice', 'noul', 'score'];
const TYPE_LABEL = { choice: 'Choice', noul: 'Noul', score: 'Score' };

// Byte-for-byte match with js/demos/rules.js::orderQuery — required so the prefill hits the
// recorded replay for presets.flip, orders[0].
function flipQuestion(flip, order) {
  const rubric = order.map((label) => `${label}: ${flip.rules[label]}`).join('  ');
  return `${flip.question_prefix}\n${rubric}`;
}

// Randomize cycles these three — all replay-backed (recorded from these exact strings; see
// data/presets.json's _note).
const RANDOM_KEYS = ['playground', 'policy', 'abstain'];

function presetToForm(key, presets) {
  const p = presets?.[key];
  if (!p) return null;
  if (key === 'abstain') {
    const variant = p.variants?.[1] ?? p.variants?.[0];
    const q = variant?.queries?.[0];
    if (!q) return null;
    return { state: p.state, question: q.question, options: q.labels.join(', '), type: q.type };
  }
  const q = p.queries?.[0];
  if (!q) return null;
  return { state: p.state, question: q.question, options: q.labels.join(', '), type: q.type };
}

export async function mount(host, ctx) {
  const presets = ctx.presets ?? {};
  const flip = presets.flip;

  host.innerHTML = '';
  const wrap = el('div', 'exhibit tryit-exhibit');

  const stateInput = document.createElement('textarea');
  stateInput.className = 'tryit-input tryit-textarea';
  stateInput.rows = 4;

  const questionInput = document.createElement('textarea');
  questionInput.className = 'tryit-input tryit-textarea';
  questionInput.rows = 1;

  // Both grow to fit their content instead of scrolling internally — textareas get a
  // UA-default overflow:auto, which becomes a real scrollbar the moment a prefilled example
  // wraps past its row count (the loan-rubric example does, on both fields).
  function autosizeOne(ta) {
    ta.style.height = 'auto';
    ta.style.height = `${ta.scrollHeight}px`;
  }
  function autosize() {
    autosizeOne(stateInput);
    autosizeOne(questionInput);
  }
  stateInput.addEventListener('input', autosize);
  questionInput.addEventListener('input', autosize);
  // Re-run once the real font has loaded (scrollHeight measured against the fallback font
  // otherwise) and on resize (reflow can change how many lines the text wraps to).
  if (typeof document !== 'undefined' && document.fonts?.ready) document.fonts.ready.then(autosize);
  window.addEventListener('resize', autosize);

  const optionsInput = document.createElement('input');
  optionsInput.type = 'text';
  optionsInput.className = 'tryit-input';

  const typeRow = el('div', 'ex-chip-row tryit-type-row');
  let type = 'choice';
  const typeBtns = TYPES.map((t) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'ex-ghost';
    b.textContent = TYPE_LABEL[t];
    b.setAttribute('aria-pressed', String(t === type));
    b.addEventListener('click', () => {
      type = t;
      typeBtns.forEach((btn, i) => btn.setAttribute('aria-pressed', String(TYPES[i] === type)));
    });
    typeRow.appendChild(b);
    return b;
  });

  const actionRow = el('div', 'tryit-action-row');
  const runBtn = document.createElement('button');
  runBtn.type = 'button';
  runBtn.className = 'ex-btn';
  runBtn.textContent = 'Run →';
  const randomBtn = document.createElement('button');
  randomBtn.type = 'button';
  randomBtn.className = 'ex-ghost';
  randomBtn.textContent = 'Randomize';
  actionRow.append(runBtn, randomBtn);

  const bars = el('div', 'tryit-bars');
  const readout = el('p', 'ex-readout');
  const notice = el('p', 'tryit-notice');

  wrap.append(
    el('label', 'ex-caption', 'State'),
    stateInput,
    el('label', 'ex-caption', 'Question'),
    questionInput,
    el('label', 'ex-caption', 'Options (comma-separated)'),
    optionsInput,
    typeRow,
    actionRow,
    bars,
    readout,
    notice
  );
  host.appendChild(wrap);

  const field = ctx.inkBars(bars, { rows: [] });

  if (flip) {
    const order = flip.orders[0];
    stateInput.value = flip.state;
    questionInput.value = flipQuestion(flip, order);
    optionsInput.value = order.join(', ');
  }
  autosize(); // stateInput is already attached (host.appendChild(wrap) above) — scrollHeight is real

  let randomIdx = 0;
  randomBtn.addEventListener('click', () => {
    const form = presetToForm(RANDOM_KEYS[randomIdx % RANDOM_KEYS.length], presets);
    randomIdx++;
    if (!form) return;
    stateInput.value = form.state;
    questionInput.value = form.question;
    optionsInput.value = form.options;
    type = form.type;
    typeBtns.forEach((btn, i) => btn.setAttribute('aria-pressed', String(TYPES[i] === type)));
    autosize();
    run();
  });

  async function run() {
    const state = stateInput.value.trim();
    const question = questionInput.value.trim();
    const labels = optionsInput.value
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    if (!state || !question || labels.length < 2) {
      readout.textContent = 'Needs a state, a question, and at least two comma-separated options.';
      return;
    }
    runBtn.disabled = true;
    const t0 = performance.now();
    const out = await ctx.decide(state, [{ type, question, labels }]);
    runBtn.disabled = false;
    if (!out) {
      field.update([], null);
      readout.textContent = '';
      notice.textContent = ctx.mode() !== 'live' ? 'Run the local server to decide on your own text.' : 'no result.';
      return;
    }
    notice.textContent = '';
    const r = out.results[0];
    const rows = labels.map((label) => ({ label, p: r.probs[label] ?? 0 }));
    field.update(rows, r.p_null ?? null);
    const ms = out.ms ?? performance.now() - t0;
    readout.textContent = `${r.argmax} (${(r.probs[r.argmax] ?? 0).toFixed(2)}) · ${ms.toFixed(0)} ms${out.device ? ' · ' + out.device : ''}`;
  }

  runBtn.addEventListener('click', run);
  await run();
}
