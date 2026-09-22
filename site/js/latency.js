// latency.js — the hosted comparison, drawn from scripts/bench_llm.py's output.
//
// The claim ledger forbade "faster than any hosted system" because no apples-to-apples run
// existed. This draws the run that now does, and it is drawn defensively on purpose:
//
//   * Both arms were timed the same way, wall clock from a client process. Our own 45 ms figure
//     is in-process on an H100 with model load excluded and does NOT appear here — the bar says
//     what the local server actually answered in, on Apple silicon, over a socket.
//   * The hosted models are shown at their lowest reasoning effort and their highest, because the
//     low number is the one anybody would ship and the high number is what the same model costs
//     when you let it think. Showing only one of the two would be picking a fight or ducking one.
//   * p10-p90 is drawn as a range behind every median. A hosted call's spread is wider than its
//     median is small, and that tail is the part that matters for a decision inside a loop.
//
// Log scale, because 66 ms and 12,774 ms do not share a linear axis legibly. Every tick is
// labelled, which is what makes a log axis honest rather than flattering.
//
//   mountLatency(host, doc)   doc = site/data/llm_latency.json

const TICKS = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000];

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10000 ? 1 : 2)} s` : `${Math.round(ms)} ms`);

// position on the log axis, 0..1, clamped to the drawn domain
export function logPos(ms, lo, hi) {
  if (!(ms > 0)) return 0;
  const t = (Math.log10(ms) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
  return Math.max(0, Math.min(1, t));
}

// "openrouter/openai/gpt-5.4-nano (low reasoning)" -> { model: 'gpt-5.4-nano', effort: 'low' }
export function parseRunKey(key) {
  const m = /^(.*?)\s*\((\w+) reasoning\)$/.exec(key) || [null, key, ''];
  const path = String(m[1]).split('/');
  return { model: path[path.length - 1], effort: m[2] || '' };
}

// The two questions the section answers, in the order it answers them. Each row is one measured
// arm; `ours` rows are the local server, everything else crossed a network.
export function rowsFor(doc) {
  const runs = Object.entries(doc?.runs ?? {});
  if (!runs.length) return [];
  const device = doc.typical_device || 'local';
  const arm = (r, k) => r.arms?.[k];
  const many = Object.keys(runs[0][1].arms || {}).find((k) => /^typical_(\d+)$/.test(k) && k !== 'typical_1');
  const m = many ? Number(many.split('_')[1]) : 8;
  // our own arms are identical across runs (same server, same state) -- take the best-sampled one
  const best = runs.reduce((a, b) => ((b[1].arms?.typical_1?.n ?? 0) > (a[1].arms?.typical_1?.n ?? 0) ? b : a))[1];

  const group = (title, oursKey, oursNote, pick) => ({
    title,
    rows: [
      { name: 'typical-small', note: `${device}, one call`, ours: true, ...arm(best, oursKey) },
      ...runs.flatMap(([key, run]) => {
        const a = pick(run);
        if (!a) return [];
        const { model, effort } = parseRunKey(key);
        return [{ name: model, note: effort ? `${effort} reasoning` : 'hosted', ...a }];
      }),
    ].filter((r) => r.median_ms != null),
    oursNote,
  });

  return [
    group('One decision on one ticket', 'typical_1', null, (r) => arm(r, 'llm_1')),
    group(`${m} questions on the same ticket`, `typical_${m}`, null, (r) => arm(r, `llm_${m}_batched`)),
    group(`${m} questions, asked one at a time`, `typical_${m}`, null, (r) => arm(r, `llm_${m}_serial`)),
  ].filter((g) => g.rows.length > 1);
}

export function mountLatency(host, doc) {
  const groups = rowsFor(doc);
  if (!host || !groups.length) return;
  host.replaceChildren();
  host.classList.add('vs');

  const all = groups.flatMap((g) => g.rows).flatMap((r) => [r.p10_ms, r.median_ms, r.p90_ms]).filter((v) => v > 0);
  const lo = Math.min(50, Math.min(...all));
  const hi = Math.max(...all) * 1.15;
  const pos = (ms) => logPos(ms, lo, hi) * 100;

  groups.forEach((g) => {
    const block = el('div', 'vs-group');
    block.appendChild(el('p', 'vs-title t-mono', g.title));
    g.rows.sort((a, b) => a.median_ms - b.median_ms).forEach((r) => {
      const row = el('div', `vs-row${r.ours ? ' is-ours' : ''}`);
      const label = el('div', 'vs-label');
      label.append(el('p', 'vs-name', r.name), el('p', 'vs-note t-mono', r.note));
      const track = el('div', 'vs-track');
      // the spread first, so the median mark sits on top of it
      const span = el('i', 'vs-span');
      span.style.left = `${pos(r.p10_ms).toFixed(2)}%`;
      span.style.width = `${Math.max(0.4, pos(r.p90_ms) - pos(r.p10_ms)).toFixed(2)}%`;
      const mark = el('b', 'vs-mark');
      mark.style.left = `${pos(r.median_ms).toFixed(2)}%`;
      track.append(span, mark);
      row.append(label, track, el('p', 'vs-val', fmtMs(r.median_ms)));
      block.appendChild(row);
    });
    host.appendChild(block);
  });

  // the axis, labelled at every decade and half-decade it actually spans
  const axis = el('div', 'vs-axis');
  const rule = el('div', 'vs-axis-rule');
  TICKS.filter((t) => t >= lo && t <= hi).forEach((t) => {
    const tick = el('span', 'vs-tick t-mono', fmtMs(t));
    tick.style.left = `${pos(t).toFixed(2)}%`;
    rule.appendChild(tick);
  });
  axis.append(rule, el('p', 'vs-axis-note t-mono', 'log scale · the bar behind each mark is p10 to p90'));
  host.appendChild(axis);

  if (doc.method) host.appendChild(el('p', 'note vs-method', doc.method));
}

export function selfTest() {
  console.assert(Math.abs(logPos(100, 10, 1000) - 0.5) < 1e-9, 'a decade sits halfway between two others');
  console.assert(logPos(5, 10, 1000) === 0 && logPos(5000, 10, 1000) === 1, 'and never leaves the axis');
  console.assert(logPos(0, 10, 1000) === 0, 'a zero reading does not become -Infinity');
  const k = parseRunKey('openrouter/openai/gpt-5.4-nano (low reasoning)');
  console.assert(k.model === 'gpt-5.4-nano' && k.effort === 'low', 'the run key gives up its model and effort');
  console.assert(parseRunKey('local/thing').model === 'thing', 'and degrades on a key without one');
  console.assert(fmtMs(66.5) === '67 ms' && fmtMs(2126) === '2.13 s' && fmtMs(12774) === '12.8 s', 'milliseconds become seconds where seconds read better');
  console.assert(rowsFor({ runs: {} }).length === 0, 'no runs, nothing drawn');
  console.log('latency.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}
