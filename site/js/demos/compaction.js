// 05 COMPACTION - does not work. 24-call tool transcript, each call scored with two Noul
// questions. P(keep) never crosses .5 on this pool, so there is no usable threshold; what's
// shown instead is the RANKING by P(keep) (AUROC .72 measured on the hand-labelled gold) -
// the honest version of this demo, not a fake "compaction in progress" animation. (The
// FAILS tag lives on the card chrome, not in this module.)
//
// TEMPLATE (mirrored verbatim in scripts/record_data_demos.py's compaction_state()):
//   state = `Task: ${task}\nTool call: ${tool} ${args}\nResult:\n${result}`
//   query[0] keep     = {type:'noul', question:KEEP_Q, labels:['no','yes']}
//     KEEP_Q (exact) = "Is this tool call's result relevant to the task? Answer yes if the
//       result is about the file, function, or test named in the task; otherwise answer no."
//   query[1] verbatim = {type:'noul', question:VERB_Q, labels:['no','yes']}
//     VERB_Q (exact) = "Does the result contain source code, a diff, or test output? Answer
//       yes if it does; otherwise answer no."

const KEEP_Q = "Is this tool call's result relevant to the task? Answer yes if the result is about the file, function, or test named in the task; otherwise answer no.";
const VERB_Q = 'Does the result contain source code, a diff, or test output? Answer yes if it does; otherwise answer no.';

function callState(task, call) {
  return `Task: ${task}\nTool call: ${call.tool} ${call.args}\nResult:\n${call.result}`;
}

// AUROC by rank-sum (Mann-Whitney U / n_pos*n_neg) - no library needed for 24 points.
function auroc(scores, golds) {
  const pos = [],
    neg = [];
  scores.forEach((s, i) => (golds[i] ? pos.push(s) : neg.push(s)));
  if (!pos.length || !neg.length) return null;
  let wins = 0;
  pos.forEach((p) => neg.forEach((n) => (wins += p > n ? 1 : p === n ? 0.5 : 0)));
  return wins / (pos.length * neg.length);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function selfTest() {
  console.assert(auroc([1, 0], [true, false]) === 1, 'auroc is 1 when the positive scores strictly higher');
  console.assert(auroc([0, 1], [true, false]) === 0, 'auroc is 0 when the positive scores strictly lower');
  console.assert(auroc([0.5, 0.5], [true, false]) === 0.5, 'auroc is 0.5 on a tie');
  console.assert(auroc([1, 1], [true, true]) === null, 'auroc is null with no negatives to rank against');
  console.log('compaction.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };

export async function mount(host, ctx) {
  const res = await fetch('data/demos/compaction.json');
  const pool = res.ok ? await res.json() : null;
  if (!pool) {
    host.innerHTML = '<p class="ex-muted">compaction pool unavailable.</p>';
    return;
  }

  host.innerHTML = '';
  const wrap = el('div', 'exhibit compaction-exhibit');

  const status = el('p', 'ex-readout', `scoring ${pool.calls.length} calls…`);
  wrap.appendChild(status);

  const register = el('div', 'compaction-register');
  wrap.appendChild(register);

  const caption = el('p', 'ex-caption', 'This model does not read code well enough to compact a coding transcript. The ranking is shown anyway.');
  wrap.appendChild(caption);

  host.appendChild(wrap);

  const scored = [];

  for (const call of pool.calls) {
    const out = await ctx.decide(callState(pool.task, call), [
      { type: 'noul', question: KEEP_Q, labels: ['no', 'yes'] },
      { type: 'noul', question: VERB_Q, labels: ['no', 'yes'] },
    ]);
    const pKeep = out?.results?.[0]?.probs?.yes ?? 0;
    const pVerb = out?.results?.[1]?.probs?.yes ?? 0;
    scored.push({ call, pKeep, pVerb });
    status.textContent = `scored ${scored.length}/${pool.calls.length}…`;
    if (ctx.mode() !== 'live') await sleep(Math.min(out?.ms ?? 15, 35));
  }

  scored.sort((a, b) => b.pKeep - a.pKeep);
  const auc = auroc(
    scored.map((s) => s.pKeep),
    scored.map((s) => s.call.keep)
  );
  const keepAcc = scored.filter((s) => (s.pKeep > 0.5) === s.call.keep).length / scored.length;
  status.textContent = `ranked by P(keep): AUROC ${auc != null ? auc.toFixed(2) : '—'} (spec: .72) · threshold-.5 accuracy ${keepAcc.toFixed(2)} (majority .542) · never crosses .5`;

  scored.forEach(({ call, pKeep }, rank) => {
    const r = el('div', 'ex-row');
    r.appendChild(el('span', 'compaction-index', String(rank + 1).padStart(2, '0')));
    const text = el('span', 'compaction-call', `${call.tool} ${call.args}`);
    text.title = `${call.tool} ${call.args}`;
    r.appendChild(text);
    const bar = el('span', 'ex-bar');
    const fill = el('span', 'ex-bar-fill' + (call.keep ? ' is-winner' : ''));
    fill.style.width = `${(pKeep * 100).toFixed(1)}%`;
    bar.appendChild(fill);
    r.appendChild(bar);
    r.appendChild(el('span', 'ex-num', pKeep.toFixed(2)));
    r.appendChild(el('span', 'compaction-mark ' + (call.keep ? 'keep' : 'drop'), call.keep ? 'KEEP' : 'DROP'));
    register.appendChild(r);
  });
}
