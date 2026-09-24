// arch.js — the read path, drawn.
//
// This section used to be a picture of a flow chart typed in box-drawing characters, with
// probabilities that came from nowhere: .82, .97, .80 were invented, and every other number on the
// site is measured. So it is a real diagram now, and it carries no probabilities at all — the two
// sections above already show what comes back. What this one owes the reader is the machine:
//
//   the ticket is encoded once → the prefix cache is kept → each question is appended to it as a
//   short suffix carrying its own candidates → a head reads the hidden state
//
// and the last stage is the claim the rest of the page rests on: nothing is ever decoded. There is
// no token to parse, constrain or retry, because no token is produced.
//
//   mountFlow(container)   -> draws, registers for resize, returns nothing
//
// Deliberately static. The instrument in the hero owns the page's only moving thing; an
// architecture diagram that pulses is decoration competing with it.

import { svgEl, svgText, registerChart, fitWidth, INK, MID, STEEL, OURS, FONT_MONO } from './charts.js';

const DESIGN_W = 1120;
const STACK_W = 620; // below this the row does not fit, so the stages stack

// The four stages, in the order the read happens. `lines` is the body of the box; `note` sits
// under it in mid-gray and says what leaves the stage.
const STAGES = [
  { tag: 'state', lines: ['ticket #7734', 'unstructured text'], note: 'read once' },
  { tag: 'encode', lines: ['Qwen3 trunk,', 'cut at ~71% depth'], note: 'one forward pass' },
  { tag: 'prefix cache', lines: ['kept, and reused by', 'every question below'], note: 'the expensive part, paid once', dashed: true },
];

// What hangs off the cache: one branch per primitive, each a short suffix plus its own candidates.
const BRANCHES = [
  { q: '+ "Which team owns this?"', k: 'choice', out: '{billing: p, …, p_null}' },
  { q: '+ "Escalate to a manager?"', k: 'noul', out: 'P(yes)' },
  { q: '+ "How urgent?"', k: 'score', out: '{0: p, …, p_null, expected}' },
];

function box(x, y, w, h, { dashed } = {}) {
  return svgEl('rect', {
    x, y, width: w, height: h, rx: 6,
    fill: 'none', stroke: dashed ? MID : STEEL, 'stroke-width': 1,
    ...(dashed ? { 'stroke-dasharray': '3 4' } : {}),
  });
}

function mono(x, y, str, { size = 12, fill = INK, anchor = 'start', weight = 400 } = {}) {
  return svgText(x, y, str, { 'font-size': size, fill, 'text-anchor': anchor, 'font-family': FONT_MONO, 'font-weight': weight });
}

// A connector with a single rounded bend, so a fan-out reads as one signal splitting rather than
// as three unrelated arrows. Straight when the ends are level.
export function elbow(x0, y0, x1, y1, r = 12) {
  if (Math.abs(y1 - y0) < 1) return `M${x0} ${y0} H${x1}`;
  const mid = x0 + (x1 - x0) * 0.42;
  const dir = y1 > y0 ? 1 : -1;
  const rr = Math.min(r, Math.abs(y1 - y0) / 2, Math.abs(mid - x0), Math.abs(x1 - mid));
  return `M${x0} ${y0} H${mid - rr} Q${mid} ${y0} ${mid} ${y0 + dir * rr} V${y1 - dir * rr} Q${mid} ${y1} ${mid + rr} ${y1} H${x1}`;
}

function arrow(svg, d) {
  svg.appendChild(svgEl('path', { d, fill: 'none', stroke: STEEL, 'stroke-width': 1 }));
}

function head(svg, x, y) {
  svg.appendChild(svgEl('path', { d: `M${x - 5} ${y - 3.2} L${x} ${y} L${x - 5} ${y + 3.2}`, fill: 'none', stroke: MID, 'stroke-width': 1 }));
}

export function mountFlow(container) {
  if (!container) return;
  const draw = () => {
    const w = fitWidth(container, DESIGN_W);
    container.innerHTML = '';
    container.style.maxWidth = `${DESIGN_W}px`;
    const svg = svgEl('svg', { width: '100%', role: 'img' });
    const title = svgEl('title');
    title.textContent = 'The read path: the state is encoded once, the prefix cache is kept, and each question is appended to it as a short suffix whose answer is read off the hidden state.';
    svg.appendChild(title);
    // the layout reports the height it used, so a stacked column is never padded out to a guess
    const h = (w < STACK_W ? column : row)(svg, w);
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    container.appendChild(svg);
  };
  draw();
  registerChart(container, draw);
}

// Wide: the three stages left to right, then the fan-out at the cache. The fan gets a long run of
// its own — bending three lines inside a 34px gap bundled them into one grey smudge.
//
// The fan used to start at a fixed y=60 while the stage row sat at 161, so the right-hand column
// floated above the left and the figure had a hole under it. The branch block is centred on the
// stage row's centre line now, and the whole drawing is measured off that one number.
const BRANCH_PITCH = 78;

function row(svg, w) {
  const pad = 2;
  const gap = 34;
  const fanGap = 96;
  const branchW = Math.min(340, w * 0.31);
  const colW = (w - pad * 2 - branchW - fanGap - gap * 2) / 3;
  const bodyY = 12 + (BRANCHES.length * BRANCH_PITCH) / 2; // the two columns share one centre line
  const bodyH = 66;
  STAGES.forEach((s, i) => {
    const x = pad + i * (colW + gap);
    svg.appendChild(mono(x, bodyY - 44, s.tag.toUpperCase(), { size: 10, fill: i === 0 ? OURS : MID, weight: 700 }));
    svg.appendChild(box(x, bodyY - 30, colW, bodyH, s));
    if (i === 0) {
      // the ticket drawn as what it becomes: a run of cells, the same ones Fig. 1 counts
      const n = 22;
      const cw = (colW - 28 - (n - 1) * 3) / n;
      for (let k = 0; k < n; k += 1) {
        svg.appendChild(svgEl('rect', { x: x + 14 + k * (cw + 3), y: bodyY - 18, width: cw, height: 12, fill: OURS, opacity: 0.85 }));
      }
      svg.appendChild(mono(x + 14, bodyY + 18, s.lines[0], { size: 11, fill: MID }));
    } else {
      s.lines.forEach((l, n2) => svg.appendChild(mono(x + 14, bodyY - 6 + n2 * 17, l, { size: 12 })));
    }
    svg.appendChild(mono(x, bodyY + 56, s.note, { size: 11, fill: MID }));
    if (i > 0) {
      arrow(svg, `M${x - gap + 5} ${bodyY + 3} H${x - 7}`);
      head(svg, x - 3, bodyY + 3);
    }
  });

  // the fan: one line out of the cache, splitting into the three typed questions
  const cacheRight = pad + 2 * (colW + gap) + colW;
  const x4 = cacheRight + fanGap;
  const fanTop = bodyY + 3 - (BRANCHES.length - 1) * BRANCH_PITCH / 2 - 20;
  BRANCHES.forEach((b, n) => {
    const y = fanTop + n * BRANCH_PITCH;
    arrow(svg, elbow(cacheRight + 5, bodyY + 3, x4 - 9, y + 20, 16));
    head(svg, x4 - 5, y + 20);
    svg.appendChild(mono(x4, y + 8, b.k.toUpperCase(), { size: 10, fill: OURS, weight: 700 }));
    svg.appendChild(mono(x4, y + 26, b.q, { size: 12 }));
    svg.appendChild(svgEl('line', { x1: x4, y1: y + 36, x2: x4 + branchW, y2: y + 36, stroke: STEEL, 'stroke-width': 1 }));
    svg.appendChild(mono(x4, y + 52, b.out, { size: 11, fill: MID }));
  });
  svg.appendChild(mono(x4, fanTop - 22, 'ASK \u00b7 READ OUT', { size: 10, fill: MID, weight: 700 }));
  return footer(svg, pad, Math.max(bodyY + 96, fanTop + BRANCHES.length * BRANCH_PITCH + 14), w - pad * 2, 11);
}

// Narrow: the same path top to bottom. A 390px column cannot hold four stages side by side, and
// shrinking the type to make it fit is the bug this whole drawing system was built to avoid.
function column(svg, w) {
  const pad = 2;
  const boxW = w - pad * 2;
  let y = 24;
  STAGES.forEach((s) => {
    svg.appendChild(mono(pad, y, s.tag.toUpperCase(), { size: 10, fill: MID, weight: 700 }));
    svg.appendChild(box(pad, y + 10, boxW, 26 + s.lines.length * 17, s));
    s.lines.forEach((l, n) => svg.appendChild(mono(pad + 14, y + 33 + n * 17, l, { size: 12 })));
    svg.appendChild(mono(pad, y + 54 + s.lines.length * 17, s.note, { size: 11, fill: MID }));
    y += 96 + s.lines.length * 17;
    arrow(svg, `M${pad + 16} ${y - 34} V${y - 20}`);
    svg.appendChild(svgEl('path', { d: `M${pad + 12.8} ${y - 25} L${pad + 16} ${y - 20} L${pad + 19.2} ${y - 25}`, fill: 'none', stroke: MID, 'stroke-width': 1 }));
  });
  svg.appendChild(mono(pad, y, 'ASK · READ OUT', { size: 10, fill: MID, weight: 700 }));
  y += 14;
  BRANCHES.forEach((b) => {
    svg.appendChild(mono(pad + 14, y + 16, b.k.toUpperCase(), { size: 10, fill: MID, weight: 700 }));
    svg.appendChild(mono(pad + 14, y + 34, b.q, { size: 12 }));
    svg.appendChild(mono(pad + 14, y + 52, b.out, { size: 11, fill: MID }));
    svg.appendChild(svgEl('line', { x1: pad, y1: y + 2, x2: pad + boxW, y2: y + 2, stroke: STEEL, 'stroke-width': 1 }));
    y += 66;
  });
  return footer(svg, pad, y + 16, boxW, 11);
}

const CLOSER = 'No token is ever decoded. The answer is a head on the hidden state, so there is nothing to parse, constrain or retry.';

// SVG text does not wrap, so on a phone the closing line used to run straight off the edge. Break
// it on words against the width the drawing actually has. Returns the height the diagram needs.
export function wrapMono(str, width, size) {
  const per = Math.max(1, Math.floor(width / (size * 0.6)));
  const out = [];
  let line = '';
  for (const word of str.split(' ')) {
    if (line && line.length + 1 + word.length > per) {
      out.push(line);
      line = word;
    } else line = line ? `${line} ${word}` : word;
  }
  if (line) out.push(line);
  return out;
}

function footer(svg, x, y, w, size) {
  svg.appendChild(svgEl('line', { x1: x, y1: y, x2: x + w, y2: y, stroke: STEEL, 'stroke-width': 1 }));
  const lines = wrapMono(CLOSER, w, size);
  lines.forEach((l, n) => svg.appendChild(mono(x, y + 20 + n * 16, l, { size, fill: MID })));
  return y + 20 + (lines.length - 1) * 16 + 10;
}

export function selfTest() {
  console.assert(elbow(0, 10, 100, 10) === 'M0 10 H100', 'level ends connect straight');
  const d = elbow(0, 0, 100, 60);
  console.assert(d.startsWith('M0 0 H') && d.includes('Q') && d.endsWith('H100'), 'a drop bends once and lands level');
  console.assert(!/NaN|undefined/.test(d), 'and never emits a broken path');
  console.assert(!/NaN/.test(elbow(0, 0, 100, 1.5)), 'including when the drop is smaller than the radius');
  console.assert(STAGES.length === 3 && BRANCHES.length === 3, 'three stages before the fan, three typed branches after it');
  const wrapped = wrapMono(CLOSER, 320, 11);
  console.assert(wrapped.length > 1 && wrapped.every((l) => l.length * 6.6 <= 320), 'the closing line breaks to fit a phone');
  console.assert(wrapped.join(' ') === CLOSER, 'and loses no words doing it');
  console.assert(wrapMono(CLOSER, 4000, 11).length === 1, 'while a wide panel keeps it on one line');
  console.log('arch.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}
