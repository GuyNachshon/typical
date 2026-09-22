// instrument.js — one state, many decisions, shown as an instrument rather than a card.
//
// The composition is the argument: a fixed state on the left, a channel carrying it across, and
// the distribution on the right as the dominant graphic. The state never changes; the question
// cycles through the typed decisions the same ticket can support, which is the whole point —
// read once, ask many.
//
// The rows are typographic, not progress bars: a hairline rule whose length is the probability,
// the winner set larger and brighter, ∅ dashed and set apart. Nothing else is chrome.
//
//   mountInstrument(el, { state, entries, decide, onLive })
//     entries: [{ q, result }]   q: {type, question, labels}
//     decide:  optional (state, [q]) => result — used to refresh a row live

const CYCLE_MS = 7000;
const FLOW_FPS = 30;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

// probability -> the width of its rule, as a share of the track. A floor keeps a near-zero row
// visible as a mark rather than nothing at all: absence and 0.00 should not look the same.
export function ruleWidth(p) {
  return Math.max(0.012, Math.min(1, p));
}

// The channel between state and decision: a hairline with a few marks travelling along it. It
// is the only moving thing in the panel, and it carries the eye left to right, which is the
// direction of the transformation.
// The connector: a line from the question being asked to the row it produced, with a few marks
// travelling along it. It is the only moving thing on the panel, and it says which question this
// readout belongs to — a line floating in the gap said nothing.
function flow(canvas, anchors = {}) {
  const ctx = canvas.getContext('2d');
  const marks = Array.from({ length: 7 }, (_, i) => ({ t: i / 7, v: 0.06 + Math.random() * 0.05 }));
  let raf = 0;
  let last = 0;
  let surge = 0; // rises when a decision lands, then decays
  let y0 = 0.5;
  let y1 = 0.5;
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;

  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (now - last < 1000 / FLOW_FPS) return;
    const dt = Math.min(64, now - last) / 1000;
    last = now;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(2, devicePixelRatio || 1);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const W = canvas.width;
    const H = canvas.height;
    // ease toward the anchors so a question change slides the connector rather than cutting it
    const ty0 = anchors.from?.() ?? 0.5;
    const ty1 = anchors.to?.() ?? 0.5;
    y0 += (ty0 - y0) * Math.min(1, dt * 7);
    y1 += (ty1 - y1) * Math.min(1, dt * 7);
    const at = (t) => {
      const e = t * t * (3 - 2 * t); // smoothstep: flat at both ends, curved in the middle
      return (y0 + (y1 - y0) * e) * H;
    };
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(240,238,235,0.16)';
    ctx.lineWidth = Math.max(1, dpr * 0.5);
    ctx.beginPath();
    for (let i = 0; i <= 24; i++) ctx[i ? 'lineTo' : 'moveTo']((i / 24) * W, at(i / 24));
    ctx.stroke();
    surge = Math.max(0, surge - dt * 1.6);
    for (const m of marks) {
      if (!still) m.t = (m.t + (m.v + surge * 0.6) * dt) % 1.2;
      if (m.t > 1) continue;
      const a = (0.2 + surge * 0.5) * (1 - Math.abs(m.t - 0.5) * 0.7);
      ctx.fillStyle = `rgba(240,238,235,${Math.max(0, a).toFixed(3)})`;
      ctx.fillRect(m.t * W, at(m.t) - dpr * 1.5, dpr * 9, dpr * 3);
    }
  }
  raf = requestAnimationFrame(frame);
  return {
    pulse() {
      surge = 1;
    },
    stop() {
      cancelAnimationFrame(raf);
    },
  };
}

export function mountInstrument(host, { state, entries, decide, onLive } = {}) {
  if (!host || !entries?.length) return { stop() {} };
  host.replaceChildren();
  host.classList.add('instrument');

  // Left: the specimen and the questions asked of it. The stack is the argument — four typed
  // decisions sitting against one state, visible at a glance instead of only over time.
  const left = el('div', 'inst-in');
  const stateHead = el('p', 'inst-tag t-mono', 'state');
  const stateText = el('p', 'inst-state-text', state);
  const qHead = el('p', 'inst-tag t-mono', 'questions asked of it');
  const list = el('div', 'inst-list');
  left.append(stateHead, stateText, qHead, list);

  // Right: the readout.
  const right = el('div', 'inst-out');
  const rows = el('div', 'inst-rows');
  const foot = el('p', 'inst-foot t-mono');
  right.append(rows, foot);

  const channel = el('div', 'inst-flow');
  const canvas = document.createElement('canvas');
  channel.appendChild(canvas);

  host.append(left, channel, right);

  const buttons = entries.map((entry, n) => {
    const b = el('button', 'inst-qbtn');
    b.type = 'button';
    b.append(el('span', 'inst-qtype t-mono', entry.q.type), el('span', 'inst-qtext', shortQuestion(entry.q.question)));
    b.addEventListener('click', () => select(n, true));
    list.appendChild(b);
    return b;
  });

  let active = 0;
  let timer = 0;
  let held = false;

  // both ends in the channel's own coordinates, so the line lands on the right things at any size
  const anchorOf = (node) => {
    if (!node) return 0.5;
    const box = canvas.getBoundingClientRect();
    if (!box.height) return 0.5;
    const r = node.getBoundingClientRect();
    return Math.max(0, Math.min(1, (r.top + r.height / 2 - box.top) / box.height));
  };
  const stream = flow(canvas, {
    from: () => anchorOf(buttons[active]),
    to: () => anchorOf(rows.querySelector('.is-win') || rows.firstElementChild),
  });


  function paint(entry, source) {
    const probs = entry.result?.probs ?? {};
    const labels = Object.keys(probs);
    const top = labels.reduce((a, b) => ((probs[b] ?? 0) > (probs[a] ?? 0) ? b : a), labels[0]);
    const list2 = [
      ...labels.map((label) => ({ label, p: probs[label] ?? 0, win: label === top })),
      { label: 'none of them', p: entry.result?.p_null ?? 0, none: true },
    ];
    rows.replaceChildren();
    list2.forEach((r, n) => {
      const row = el('div', `inst-row${r.win ? ' is-win' : ''}${r.none ? ' is-none' : ''}`);
      row.style.setProperty('--n', String(n));
      const name = el('span', 'inst-label', r.label);
      const track = el('span', 'inst-track');
      const rule = el('i', 'inst-rule');
      rule.style.width = `${(ruleWidth(r.p) * 100).toFixed(1)}%`;
      track.appendChild(rule);
      const val = el('span', 'inst-value', r.p.toFixed(2));
      row.append(name, track, val);
      rows.appendChild(row);
    });
    foot.textContent = `${source} · ${Math.round(entry.result?.ms ?? 0)} ms · the state was read once`;
    stream.pulse();
  }

  async function select(n, fromClick) {
    active = n % entries.length;
    buttons.forEach((b, k) => b.classList.toggle('is-on', k === active));
    const entry = entries[active];
    paint(entry, entry.live ? 'live' : 'recorded');
    if (fromClick) restart();
    if (!decide) return;
    try {
      const res = await decide(state, [entry.q]);
      const r = res?.results?.[0];
      if (r && active === n % entries.length) {
        entry.result = { ...r, ms: res.ms };
        entry.live = true;
        paint(entry, 'live');
        onLive?.(res, state, [entry.q]);
      }
    } catch {}
  }

  function restart() {
    clearInterval(timer);
    timer = setInterval(() => { if (!held) select(active + 1); }, CYCLE_MS);
  }

  select(0);
  restart();
  host.addEventListener('pointerenter', () => { held = true; });
  host.addEventListener('pointerleave', () => { held = false; });

  return {
    stop() {
      clearInterval(timer);
      stream.stop();
    },
  };
}

// The question list carries the sense of each question, not its full instruction text.
export function shortQuestion(q) {
  const first = String(q).split(/(?<=\?)\s/)[0].trim();
  return first.length > 46 ? `${first.slice(0, 45)}…` : first;
}

export function selfTest() {
  console.assert(ruleWidth(0) > 0, 'a zero row still leaves a mark');
  console.assert(ruleWidth(1) === 1 && ruleWidth(2) === 1, 'the rule never exceeds its track');
  console.assert(ruleWidth(0.5) === 0.5, 'and is linear in between');
  console.assert(shortQuestion('Which team should own this ticket? Answer with one of the teams.') === 'Which team should own this ticket?', 'the list shows the question, not the instructions after it');
  console.assert(shortQuestion('x'.repeat(60)).endsWith('…'), 'and never runs past its column');
  console.log('instrument.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}
