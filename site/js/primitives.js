// primitives.js — the program, not another readout.
//
// The section above this one already shows a state being read and a distribution coming back. A
// second panel with the same ticket, the same questions and the same rows says nothing new, so
// this one changes the subject: the dominant object here is ordinary Python, and the model's
// numbers are annotations on the lines that called for them.
//
// The claim it makes, which the instrument above cannot: a distribution is only interesting
// because a branch depends on it. Take "support" out of the answer space and ∅ passes .5, so the
// program stops paging anyone and calls a human instead. The line that runs changes.
//
// The calls are the real ones from inference/typical/core.py, returning what it actually returns:
// plain dicts for choice and score, a bare float for noul. That also names the three primitives
// where a reader meets them, instead of in a panel that defines them first.
//
//   mountProgram(host, { state, questions, resultFor, decide })
//     resultFor(q) -> recorded result or undefined      decide(state, [q...]) -> live results

import { fmtProb as fmt } from './api.js';

const ADDABLE = 'send a replacement';

const CALLS = [
  { key: 'team', type: 'choice', code: ['team', '= m.choice(ticket, "Which team owns this?", TEAMS)'] },
  { key: 'urgency', type: 'score', code: ['urgency', '= m.score(ticket, "How urgent?", LEVELS)'] },
  { key: 'escalate', type: 'noul', code: ['escalate', '= m.noul(ticket, "Escalate to a manager?")'] },
  { key: null, code: ['owner', '= max(TEAMS, key=team.get)'] },
];

const PROGRAM = [
  { cond: 'if team["p_null"] > .5:', body: 'human_triage()', reads: ['none'], test: (v) => v.none > 0.5 },
  { cond: 'elif escalate > .9 and urgency["expected"] > 2.0:', body: 'page(owner)', reads: ['escalate', 'urgency'], test: (v) => v.escalate > 0.9 && v.urgency > 2 },
  { cond: 'else:', body: 'queue(owner)', reads: [], test: () => true },
];

// urgency is a position on a scale, so it keeps its units digit; probabilities drop the leading zero
const fmtRead = (key, p) => (key === 'urgency' ? p.toFixed(2) : fmt(p));

// which line of PROGRAM the current readings select, or -1 before anything has run
export function branchOf(v) {
  return v ? PROGRAM.findIndex((b) => b.test(v)) : -1;
}

// what a call line shows next to itself once it has an answer: the value the code below reads,
// not the whole distribution — that is one click away, on the only call where it can be changed
export function readOf(type, result) {
  if (!result) return '';
  if (type === 'noul') return fmt(result.probs?.yes ?? 0);
  if (type === 'score') return (result.expected ?? 0).toFixed(2);
  const probs = result.probs ?? {};
  const top = Object.keys(probs).reduce((a, b) => (probs[b] > probs[a] ? b : a), Object.keys(probs)[0]);
  return top ? `"${top}" ${fmt(probs[top])}` : '';
}

// A line under the expanded choice: taking the winner out of the answer space moves ∅ far enough
// to be the point of the whole section, and it has to be readable.
export function nullNote(pNull, removedWinner) {
  if (removedWinner && pNull > 0.4) return 'The answer you removed is gone, so ∅ takes the weight it had — past the threshold on the first line, so the program asks a human instead of paging one.';
  if (pNull > 0.4) return 'None of the options you passed fit this ticket.';
  return '';
}

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

export function mountProgram(host, { state, questions, resultFor, decide } = {}) {
  if (!host || !questions?.length) return { stop() {} };
  host.replaceChildren();
  host.classList.add('program');

  // one call per primitive, carrying the question it stands for and whatever came back for it
  const calls = CALLS.map((c) => ({
    ...c,
    q: c.key ? { ...questions.find((q) => q.type === c.type), labels: [...(questions.find((q) => q.type === c.type)?.labels || [])] } : null,
    result: null,
    removed: [],
    droppedWinner: false,
  }));
  if (calls.some((c) => c.key && !c.q?.question)) return { stop() {} };

  const head = el('div', 'program-head');
  const runBtn = el('button', 'btn dark', 'Run typical');
  runBtn.type = 'button';
  const foot = el('p', 'program-foot t-mono');
  head.append(el('p', 'program-tag t-mono', `your program · ${state.split(':')[0]}`), runBtn);
  const code = el('div', 'program-code');
  host.append(head, code, foot);

  let open = 'team'; // the choice is expanded by default: it is the one you can change
  // Where each rule was last drawn, so a redraw starts from the old value and the CSS width
  // transition has something to run from. Without this every rule is a brand-new node whose width
  // is already final, and the distribution snaps instead of moving — which is the one thing this
  // panel exists to show.
  const lastWidth = new Map();

  function readings() {
    const by = (k) => calls.find((c) => c.key === k)?.result;
    const [team, urgency, escalate] = [by('team'), by('urgency'), by('escalate')];
    if (!team || !urgency || !escalate) return null;
    return { none: team.p_null ?? 0, urgency: urgency.expected ?? 0, escalate: escalate.probs?.yes ?? 0 };
  }

  function line(cls, text) {
    return el('div', `program-line${cls ? ` ${cls}` : ''}`, text);
  }

  function render() {
    const v = readings();
    const taken = branchOf(v);
    code.replaceChildren();

    calls.forEach((c) => {
      const row = line('is-call');
      const src = el('code', 'program-src');
      src.append(el('b', null, c.code[0]), document.createTextNode(` ${c.code[1]}`));
      row.append(src);
      if (c.key) {
        const read = el('span', 'program-read t-mono', readOf(c.type, c.result));
        row.appendChild(read);
        if (c.type === 'choice') {
          row.classList.add('is-openable');
          row.tabIndex = 0;
          const toggle = () => { open = open === c.key ? null : c.key; render(); };
          row.addEventListener('click', toggle);
          row.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
        }
      }
      code.appendChild(row);
      if (c.key === open && c.type === 'choice') code.appendChild(expand(c));
    });

    code.appendChild(line('is-gap', ' '));
    PROGRAM.forEach((b, n) => {
      const on = n === taken ? ' is-taken' : '';
      const cond = line(`is-branch${on}`, b.cond);
      // the numbers the condition tested, beside the condition — so ".72 > .5" is something the
      // reader sees rather than something the prose has to assert
      if (v && b.reads.length) cond.appendChild(el('span', 'program-read t-mono', b.reads.map((k) => fmtRead(k, v[k])).join(' · ')));
      code.append(cond, line(`is-branch is-body${on}`, `    ${b.body}`));
    });
    code.classList.toggle('is-live', taken >= 0);
  }

  // The answer space, opened out: every option with its share, ∅ below the rule, and an × that
  // takes an option out. This is the only editable thing on the panel, and editing it is the
  // demonstration — the distribution moves, and two lines down a different branch lights up.
  function expand(c) {
    const wrap = el('div', 'program-expand');
    const probs = c.result?.probs || {};
    const top = Object.keys(probs).reduce((a, b) => (probs[b] > probs[a] ? b : a), Object.keys(probs)[0]);
    c.q.labels.forEach((label) => {
      const p = probs[label];
      const row = el('div', `program-row${label === top && c.result ? ' is-win' : ''}`);
      const track = el('span', 'program-track');
      const rule = el('i');
      const target = c.result ? `${Math.max(1, (p ?? 0) * 100).toFixed(1)}%` : '0%';
      rule.style.width = lastWidth.get(label) ?? '0%';
      track.appendChild(rule);
      requestAnimationFrame(() => {
        rule.style.width = target;
        lastWidth.set(label, target);
      });
      const drop = el('button', 'program-drop', '×');
      drop.type = 'button';
      drop.title = `take "${label}" out of the answer space`;
      drop.addEventListener('click', (e) => {
        e.stopPropagation();
        c.droppedWinner = label === top;
        c.removed.push(label);
        c.q.labels = c.q.labels.filter((l) => l !== label);
        resolve([c], 'the answer space changed');
      });
      row.append(el('span', 'program-label', label), track, el('span', 'program-val t-mono', c.result ? fmt(p ?? 0) : ''), drop);
      wrap.appendChild(row);
    });
    const none = el('div', 'program-row is-none');
    none.append(
      el('span', 'program-label', '∅ none of them'),
      el('span', 'program-track'),
      el('span', 'program-val t-mono', c.result ? fmt(c.result.p_null ?? 0) : ''),
    );
    wrap.appendChild(none);
    const note = c.result ? nullNote(c.result.p_null ?? 0, c.droppedWinner) : '';
    if (note) wrap.appendChild(el('p', 'program-note', note));
    if (c.result && c.removed.length) {
      wrap.appendChild(editBtn(`+ put back ${c.removed[c.removed.length - 1]}`, () => {
        const label = c.removed.pop();
        const order = questions.find((q) => q.type === 'choice').labels;
        c.q.labels = [...order, ADDABLE].filter((l) => l === label || c.q.labels.includes(l));
        c.droppedWinner = false;
        resolve([c], 'the answer space changed');
      }));
    } else if (c.result && !c.q.labels.includes(ADDABLE)) {
      wrap.appendChild(editBtn(`+ ${ADDABLE}`, () => {
        c.q.labels = [...c.q.labels, ADDABLE];
        resolve([c], 'the answer space changed');
      }));
    }
    return wrap;
  }

  function editBtn(text, onClick) {
    const b = el('button', 'program-add', text);
    b.type = 'button';
    b.addEventListener('click', (e) => { e.stopPropagation(); onClick(); });
    return b;
  }

  // Resolve a set of calls. Live, that is one request carrying every question, which is the honest
  // version of "one read"; offline it is the recordings, revealed in the same frame.
  async function resolve(set, label) {
    host.classList.add('is-running');
    const started = performance.now();
    let source = 'recorded';
    let ms = 0;
    let results = set.map((c) => resultFor(c.q));
    if (decide) {
      try {
        const res = await decide(state, set.map((c) => c.q));
        if (res?.results?.length === set.length) {
          results = res.results;
          source = 'live';
          ms = res.ms;
        }
      } catch {}
    }
    if (source === 'recorded') {
      // let a replay breathe for a beat, so the fill still reads as a single event
      await new Promise((r) => setTimeout(r, Math.max(0, 260 - (performance.now() - started))));
      ms = set.reduce((a, c) => a + (resultFor(c.q)?.ms ?? 0), 0);
    }
    set.forEach((c, n) => { c.result = results[n] || c.result; });
    render();
    host.classList.remove('is-running');
    runBtn.textContent = 'Run again';
    foot.textContent = `${label} · ${source} · ${Math.round(ms)} ms`;
  }

  render();
  runBtn.addEventListener('click', () => resolve(calls.filter((c) => c.key), 'three calls, one read of the ticket'));

  // The panel rendered blank until someone pressed the button, which reads as broken rather than
  // as an invitation. It runs itself once when it comes into view; the button then says "Run again"
  // and still does everything it did.
  let armed = true;
  const io = new IntersectionObserver((entries) => {
    if (!armed || !entries.some((e) => e.isIntersecting)) return;
    armed = false;
    io.disconnect();
    resolve(calls.filter((c) => c.key), 'three calls, one read of the ticket');
  }, { threshold: 0.35 });
  io.observe(host);

  return { stop() { io.disconnect(); } };
}

export function selfTest() {
  console.assert(readOf('choice', { probs: { a: 0.6, b: 0.4 }, p_null: 0.1 }) === '"a" .60', 'a choice reads as its winner');
  console.assert(readOf('score', { expected: 2.5009 }) === '2.50', 'a score reads as its expected index');
  console.assert(readOf('noul', { probs: { yes: 0.998 } }) === '.998', 'a noul reads as P(yes), and never rounds to certainty');
  console.assert(readOf('choice', null) === '', 'and nothing reads as nothing');
  // the recorded numbers, before and after "support" is taken out of the answer space
  console.assert(branchOf({ none: 0.083, escalate: 0.998, urgency: 2.5 }) === 1, 'the ticket pages someone');
  console.assert(branchOf({ none: 0.722, escalate: 0.998, urgency: 2.5 }) === 0, 'and drops to a human once ∅ carries it');
  console.assert(branchOf({ none: 0.083, escalate: 0.2, urgency: 2.5 }) === 2, 'an unescalated ticket just queues');
  console.assert(branchOf(null) === -1, 'nothing is taken before anything has run');
  console.assert(nullNote(0.72, true).includes('asks a human'), 'dropping the winner names what changed in the code');
  console.assert(nullNote(0.08, false) === '', 'and stays quiet otherwise');
  console.log('primitives.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}
