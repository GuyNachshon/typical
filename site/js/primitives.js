// primitives.js — one state, three questions, all resolving at once.
//
// The section's claim is "not tokens, probabilities". A tabbed version of this said it and then
// undercut it: tabs imply sequence, and sequence is what a generating model does. So the three
// questions sit on the board together and fill in the same instant, which is the difference
// being claimed, demonstrated.
//
// The interactions each teach one property, and none of them are decorative:
//   run              all three resolve together, off one read of the state
//   ask another      a fourth question resolves alone — the state was kept, not recomputed
//   drop a candidate the distribution recomputes, and taking the winner out sends ∅ up
//   add a candidate  the answer space is yours, defined per call
//
// Each question renders in its own idiom, because that is what distinguishes the three types:
// a distribution for choice, one number on an axis for noul, an ordered scale for score.
//
//   mountBoard(host, { state, questions, extra, resultFor, decide })
//     resultFor(q) -> recorded result or undefined      decide(state, [q...]) -> live results

const ADDABLE = 'send a replacement';

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

const fmt = (p) => p.toFixed(2).replace(/^0/, '');

// The short form of a question: the sentence being asked, without the instructions after it.
export function askOf(q) {
  return String(q.question).split(/(?<=\?)\s/)[0].trim();
}

// A line under the block, not text in the ∅ row: taking the winner out of the answer space moves
// that number far enough to be the point of the whole section, and it has to be readable.
export function nullNote(pNull, removedWinner) {
  if (removedWinner && pNull > 0.4) return 'The answer you removed is gone, so ∅ takes the weight it had.';
  if (pNull > 0.4) return 'None of the options you passed fit this ticket.';
  return '';
}

export function mountBoard(host, { state, questions, extra, resultFor, decide } = {}) {
  if (!host || !questions?.length) return { stop() {} };
  host.replaceChildren();
  host.classList.add('board');

  const left = el('div', 'board-state');
  left.append(el('p', 'board-tag t-mono', 'state'), el('p', 'board-state-text', state));
  const cached = el('p', 'board-cached t-mono', 'state cached');
  left.append(cached);

  const right = el('div', 'board-questions');
  const controls = el('div', 'board-controls');
  const runBtn = el('button', 'btn dark board-run', 'Run typical');
  runBtn.type = 'button';
  const askBtn = el('button', 'btn outline board-ask', '+ ask another');
  askBtn.type = 'button';
  askBtn.hidden = true;
  const foot = el('p', 'board-foot t-mono');
  controls.append(runBtn, askBtn, foot);
  host.append(left, right, controls);

  // one block per question, built empty; running fills them
  const blocks = questions.map((q) => makeBlock(q));
  blocks.forEach((b) => right.appendChild(b.node));
  let asked = false;

  function makeBlock(q) {
    const node = el('div', 'board-q');
    const head = el('div', 'board-qhead');
    head.append(el('p', 'board-ask-text', askOf(q)), el('span', 'board-type t-mono', q.type));
    const out = el('div', 'board-out');
    node.append(head, out);
    const block = { node, out, q: { ...q, labels: [...(q.labels || [])] }, result: null, removed: [], droppedWinner: false };
    render(block);
    return block;
  }

  function render(block) {
    const { q, result } = block;
    block.out.replaceChildren();
    if (q.type === 'noul') renderNoul(block);
    else if (q.type === 'score') renderScore(block);
    else renderChoice(block);
    block.out.classList.toggle('is-empty', !result);
  }

  function renderChoice(block) {
    const { q, result } = block;
    const probs = result?.probs || {};
    const top = Object.keys(probs).reduce((a, b) => (probs[b] > probs[a] ? b : a), Object.keys(probs)[0]);
    q.labels.forEach((label) => {
      const p = probs[label];
      const row = el('div', `board-row${label === top && result ? ' is-win' : ''}`);
      const name = el('span', 'board-label', label);
      const track = el('span', 'board-track');
      const rule = el('i');
      rule.style.width = result ? `${Math.max(1, (p ?? 0) * 100).toFixed(1)}%` : '0%';
      track.appendChild(rule);
      const val = el('span', 'board-val', result ? fmt(p ?? 0) : '');
      const drop = el('button', 'board-drop', '×');
      drop.type = 'button';
      drop.title = `take "${label}" out of the answer space`;
      drop.addEventListener('click', (e) => {
        e.stopPropagation();
        block.droppedWinner = label === top;
        block.removed.push(label);
        block.q.labels = block.q.labels.filter((l) => l !== label);
        resolve([block], 'one question');
      });
      row.append(name, track, val, drop);
      block.out.appendChild(row);
    });
    const none = el('div', 'board-row is-none');
    none.append(
      el('span', 'board-label', '∅ none of them'),
      el('span', 'board-track'),
      el('span', 'board-val', result ? fmt(result.p_null ?? 0) : ''),
    );
    block.out.appendChild(none);
    const note = result ? nullNote(result.p_null ?? 0, block.droppedWinner) : '';
    if (note) block.out.appendChild(el('p', 'board-note', note));
    if (result && block.removed.length) {
      const back = el('button', 'board-add', `+ put back ${block.removed[block.removed.length - 1]}`);
      back.type = 'button';
      back.addEventListener('click', () => {
        const label = block.removed.pop();
        block.q.labels = [...questions[0].labels, ...(block.q.labels.includes(ADDABLE) ? [ADDABLE] : [])].filter(
          (l) => l === label || block.q.labels.includes(l),
        );
        block.droppedWinner = false;
        resolve([block], 'one question');
      });
      block.out.appendChild(back);
    } else if (result && !block.q.labels.includes(ADDABLE)) {
      const add = el('button', 'board-add', `+ ${ADDABLE}`);
      add.type = 'button';
      add.addEventListener('click', () => {
        block.q.labels = [...block.q.labels, ADDABLE];
        resolve([block], 'one question');
      });
      block.out.appendChild(add);
    }
  }

  function renderNoul(block) {
    const p = block.result?.probs?.yes;
    const wrap = el('div', 'board-noul');
    const value = el('p', 'board-noul-val', block.result ? fmt(p ?? 0) : '');
    const axis = el('div', 'board-noul-axis');
    const fill = el('i');
    fill.style.width = block.result ? `${((p ?? 0) * 100).toFixed(1)}%` : '0%';
    axis.appendChild(fill);
    const ends = el('div', 'board-noul-ends t-mono');
    ends.append(el('span', null, 'no'), el('span', null, 'yes'));
    wrap.append(value, axis, ends);
    block.out.appendChild(wrap);
  }

  function renderScore(block) {
    const { result } = block;
    const levels = block.q.labels || [];
    const wrap = el('div', 'board-score');
    const line = el('div', 'board-score-line');
    levels.forEach((lv) => {
      const stop = el('div', 'board-stop');
      const dot = el('i');
      const p = result?.probs?.[lv] ?? 0;
      dot.style.transform = `scale(${result ? (0.3 + p * 1.7).toFixed(2) : 0.3})`;
      stop.append(dot, el('span', 'board-stop-label', lv));
      line.appendChild(stop);
    });
    wrap.appendChild(line);
    if (result) {
      const e = result.expected ?? 0;
      const caret = el('div', 'board-caret');
      caret.style.left = `${((e / Math.max(1, levels.length - 1)) * 100).toFixed(1)}%`;
      caret.appendChild(el('span', 'board-caret-val t-mono', e.toFixed(2)));
      wrap.appendChild(caret);
    }
    block.out.appendChild(wrap);
  }

  // Resolve a set of blocks together. Live, that is one call carrying every question, which is
  // the honest version of "all at once"; offline it is the recordings, revealed in one frame.
  async function resolve(set, label) {
    host.classList.add('is-running');
    const started = performance.now();
    let source = 'recorded';
    let ms = 0;
    let results = set.map((b) => resultFor(b.q));
    if (decide) {
      try {
        const res = await decide(state, set.map((b) => b.q));
        if (res?.results?.length === set.length) {
          results = res.results;
          source = 'live';
          ms = res.ms;
        }
      } catch {}
    }
    if (source === 'recorded') {
      ms = Math.round(performance.now() - started);
      // let a replay breathe for a beat, so the fill still reads as a single event
      await new Promise((r) => setTimeout(r, Math.max(0, 260 - ms)));
      ms = set.reduce((a, b) => a + (resultFor(b.q)?.ms ?? 0), 0);
    }
    set.forEach((b, n) => {
      b.result = results[n] || b.result;
      render(b);
    });
    host.classList.remove('is-running');
    host.classList.add('has-run');
    askBtn.hidden = asked || !extra;
    foot.textContent = `${label} · ${source} · ${Math.round(ms)} ms · one read of the state`;
    cached.classList.add('is-on');
  }

  runBtn.addEventListener('click', () => {
    runBtn.textContent = 'Run again';
    resolve(blocks, 'three questions, one pass');
  });
  askBtn.addEventListener('click', () => {
    if (asked || !extra) return;
    asked = true;
    askBtn.hidden = true;
    const block = makeBlock(extra);
    block.node.classList.add('is-new');
    right.appendChild(block.node);
    blocks.push(block);
    resolve([block], 'a fourth question, on the state already read');
  });

  return { stop() {} };
}

export function selfTest() {
  console.assert(askOf({ question: 'Which team should own this ticket? Answer with one.' }) === 'Which team should own this ticket?', 'the board asks the question, not the instructions');
  console.assert(nullNote(0.72, true).includes('gone'), 'dropping the winner names why ∅ moved');
  console.assert(nullNote(0.08, false) === '', 'and stays quiet otherwise');
  console.assert(fmt(0.083) === '.08', 'two places, no leading zero');
  console.log('primitives.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}
