// showdown.js -- the same twenty questions, asked of everyone, replayed at the speed they arrived.
//
// Two panes, started together. On the left the whole response lands at once, because Typical reads
// one state and scores twenty suffixes against it in a single pass. On the right the answers appear
// one at a time, because a generative model writes them one at a time.
//
// Everything here is a REPLAY of a recording, not a live call -- a browser cannot hold an API key.
// The recording is real: scripts/bench_llm.py --capture streams the hosted request and timestamps
// the instant each answer's value is complete in the token buffer, so a line appears here exactly
// when the model finished writing it. Nothing about the pacing is invented.
//
//   mountShowdown(host, doc)   doc = site/data/showdown.json

const secs = (ms) => `${(ms / 1000).toFixed(3)}s`;

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

// Inside a JSON pane a probability has to stay valid JSON, so this is not fmtProb: no bare ".84",
// and no "<.001". The one rule they share is the one that matters -- two places is the reading
// width, and the only reason to spend more is that two would print a certainty the model does not
// have. A model that said .998 must never appear on this page as 1.
export function jnum(p) {
  if (p === 0 || p === 1) return String(p);
  const s = p.toFixed(2);
  if (s === '1.00') return p.toFixed(3);            // .998 is not 1
  if (s === '0.00') return String(Number(p.toPrecision(2)));  // and .0003 is not 0
  return s;
}

// A choice ranks, so the reader sees where the mass went. A score is ordered and keeps its own
// order, because the thing worth seeing in an ordered distribution is that it decays away from the
// answer -- re-sorting 0..3 by probability destroys exactly that.
export function oursLine(q, a) {
  const entries = Object.entries(a.probs);
  if (q.type !== 'score') entries.sort((x, y) => y[1] - x[1]);
  const body = entries.map(([k, v]) => `"${k}": ${jnum(v)}`).concat(`"p_null": ${jnum(a.p_null)}`);
  return `${JSON.stringify(q.question)}: {${body.join(', ')}},`;
}

export function hostedLine(q, a) {
  return `${JSON.stringify(q.question)}: ${JSON.stringify(a.pick)},`;
}

// How often a model asserted it was sure. Typical's number is its head's probability; the hosted
// one is a field the schema asked it to fill in. They are not the same measurement and the panes
// label them differently, but "how often did you claim certainty" is answerable of both.
export function certainty(answers, at = 0.9) {
  const p = answers.map((a) => (a.probs ? a.probs[a.pick] : a.stated_confidence)).filter((v) => v != null);
  return { n: p.length, sure: p.filter((v) => v >= at).length, max: p.filter((v) => v >= 0.999).length };
}

function pane(tagText) {
  const box = el('div', 'pane');
  const body = el('div', 'pane-body');
  const foot = el('p', 'pane-foot t-mono', '');
  box.append(el('p', 'pane-tag t-mono', tagText), body, foot);
  return { box, body, foot, setFoot: (t) => { foot.textContent = t; } };
}

export function mountShowdown(host, doc) {
  const qs = doc?.questions ?? [];
  const models = Object.entries(doc?.models ?? {});
  const ours = models.find(([, m]) => m.ours)?.[1];
  const hosted = models.filter(([, m]) => !m.ours);
  if (!host || !qs.length || !ours || !hosted.length) return;
  host.replaceChildren();
  host.classList.add('show');

  const head = el('div', 'show-head');
  const replay = el('button', 'btn dark', 'Ask all 20');
  replay.type = 'button';
  const picker = el('div', 'show-pick');
  head.append(el('p', 'show-eyebrow t-mono', `${qs.length} questions · one request each · started together`), picker, replay);
  host.appendChild(head);

  const panes = el('div', 'show-panes');
  const L = pane('typical');
  const R = pane('hosted');
  panes.append(L.box, R.box);
  L.box.classList.add('is-ours');
  host.appendChild(panes);

  let pick = 0;
  const buttons = hosted.map(([key], i) => {
    const name = String(key.split(' (')[0]).split('/').pop();
    const b = el('button', 'show-pick-btn t-mono', name);
    b.type = 'button';
    b.addEventListener('click', () => { pick = i; render(); run(); });
    picker.appendChild(b);
    return b;
  });

  const oursLines = qs.map((q, i) => oursLine(q, ours.answers[i]));
  let raf = 0;
  let rows = { L: [], R: [] };

  // The pane is built once at full height with every line already in place and hidden, so a line
  // landing never reflows the page. A terminal that grows while you read it is a different thing
  // from one filling in, and the second is what was measured.
  function fill(target, header, lines) {
    target.body.replaceChildren();
    target.body.append(el('p', 'pane-cmd', header));
    const out = lines.map((t) => {
      const n = el('p', t === '{' || t === '}' ? 'pane-line is-brace' : 'pane-line', t);
      target.body.appendChild(n);
      return n;
    });
    return out;
  }

  function render() {
    const [key, m] = hosted[pick];
    buttons.forEach((b, i) => b.classList.toggle('is-on', i === pick));
    const name = String(key.split(' (')[0]).split('/').pop();
    rows.L = fill(L, `$ typical.ask(ticket, questions)   # ${qs.length} questions, one pass`, ['{', ...oursLines, '}']);
    rows.R = fill(R, `$ POST /chat/completions            # ${qs.length} questions, one request`, ['{', ...qs.map((q, i) => hostedLine(q, m.answers[i])), '}']);
    L.box.dataset.model = `typical-small · ${doc.typical_device || ours.device || 'local'}`;
    R.box.dataset.model = `${name} · ${(key.match(/\((\w+) reasoning\)/) || [, ''])[1]} reasoning`;
  }

  function paint(target, lines, shown, done, ms, t) {
    lines.forEach((n, i) => n.classList.toggle('is-in', i <= shown));
    // an empty pane for five seconds reads as broken. The wait is the measurement, so it gets a
    // running clock rather than a spinner, and the clock stops where the recording stopped.
    target.setFoot(done ? `completed in ${secs(ms)}` : `elapsed ${secs(Math.max(0, t))}`);
    target.box.classList.toggle('is-done', done);
  }

  function run() {
    cancelAnimationFrame(raf);
    const m = hosted[pick][1];
    // timeline[i] is when answer i finished arriving; +1 for the opening brace above them
    const at = new Array(qs.length).fill(m.ms);
    (m.timeline || []).forEach((t) => { if (t.q < at.length) at[t.q] = t.t_ms; });
    const span = Math.max(ours.ms, m.ms);
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (still) {
      paint(L, rows.L, rows.L.length, true, ours.ms, ours.ms);
      paint(R, rows.R, rows.R.length, true, m.ms, m.ms);
      return;
    }
    rows.L.concat(rows.R).forEach((n) => n.classList.remove('is-in'));
    const t0 = performance.now();
    const step = (now) => {
      const t = now - t0;
      // ours is one call: the brace and every line land together, because they did
      paint(L, rows.L, t >= ours.ms ? rows.L.length : -1, t >= ours.ms, ours.ms, t);
      // nothing is drawn in the hosted pane before its first byte, including the brace
      const first = m.first_token_ms ?? m.ms;
      let shown = t >= first ? 0 : -1;
      at.forEach((v) => { if (t >= v) shown += 1; });
      const done = t >= m.ms;
      const waiting = t < first;
      rows.R[0].textContent = waiting ? 'waiting for first token…' : '{';
      rows.R[0].classList.toggle('is-wait', waiting);
      rows.R[0].classList.toggle('is-brace', !waiting);
      paint(R, rows.R, done ? rows.R.length : (waiting ? 0 : shown), done, m.ms, t);
      if (t < span) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  }

  render();
  replay.addEventListener('click', run);
  const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); run(); } }, { threshold: 0.25 });
  io.observe(host);
  return { run, stop() { cancelAnimationFrame(raf); io.disconnect(); } };
}

export function selfTest() {
  console.assert(jnum(0.9984) === '0.998', 'a near-certainty is never printed as 1');
  console.assert(jnum(1) === '1' && jnum(0) === '0', 'an exact bound prints as itself');
  console.assert(jnum(0.84) === '0.84' && jnum(0.0003) === '0.0003', 'two places, more only when two would read as a bound');
  const q = { type: 'choice', question: 'Which team should own this ticket?' };
  const a = { pick: 'support', probs: { platform: 0.3, support: 0.46, billing: 0.2, success: 0.04 }, p_null: 0.02 };
  const line = oursLine(q, a);
  console.assert(line.indexOf('"support": 0.46') < line.indexOf('"platform": 0.3'), 'a choice ranks');
  console.assert(line.endsWith('"p_null": 0.02},'), 'the abstention is the last field, always present');
  const ord = oursLine({ type: 'score', question: 'x' }, { pick: '3', probs: { 0: 0.02, 1: 0.11, 2: 0.23, 3: 0.64 }, p_null: 0.001 });
  console.assert(ord.indexOf('"0"') < ord.indexOf('"3"'), 'ordered levels keep their order');
  console.assert(hostedLine(q, { pick: 'platform' }) === '"Which team should own this ticket?": "platform",', 'the hosted pane prints the label and nothing else');
  const c = certainty([{ pick: 'a', probs: { a: 0.46 } }, { pick: 'a', probs: { a: 0.99 } }]);
  console.assert(c.n === 2 && c.sure === 1, 'certainty counts what cleared the bar');
  console.assert(certainty([{ pick: 'a', stated_confidence: 1 }]).max === 1, 'a stated 1.00 is counted on either side');
  console.assert(secs(477.7) === '0.478s', 'the clock reads in seconds, three places');
  console.log('showdown.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) selfTest();
