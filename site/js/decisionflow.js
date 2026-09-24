// decisionflow.js — the formalism in §formal, drawn and made to move.
//
// The section states a decision as a tuple (S, Q, C, type), takes one pass through a truncated
// backbone, reads two states off the tap, scores them bilinearly and gates the result. That is a
// pipeline, and prose reads it as five disconnected equations. This draws it as one left-to-right
// run and animates the run, so the order of operations is visible before the maths is.
//
//   mountDecisionFlow(container) -> draws, registers for resize, returns nothing
//
// Two layout rules do most of the work:
//
//   * One spine. Everything that carries the decision forward sits on a single horizontal line at
//     SPINE_Y — the arrows, the tap, the score box, the fork between the two states. The stack is
//     the one thing deliberately off-centre about it, because the tap is at layer 20 of 28 and the
//     asymmetry of the stack around the spine *is* that fact.
//   * One key column. Chip values all start at the same x rather than after their own label, so
//     the four request rows read as a table instead of as ragged text.
//
// On animation: arch.js argues an architecture diagram should stay still so it does not compete
// with the hero. That holds for a diagram whose job is to be referenced. This one's job is to show
// a sequence, and the sequence is the content. So it moves — but it moves the way a circuit does,
// by drawing along its own connectors, not by sliding a dot along them. A travelling dot reads as
// a cartoon; a line that draws itself in the direction of travel reads as flow.
//
// It runs only while on screen (IntersectionObserver), and under prefers-reduced-motion it renders
// its finished state and never builds a timeline. Nothing here carries a probability: the bars at
// the right are labelled as shape, for the same reason arch.js refuses them.

import { svgEl, svgText, registerChart, fitWidth, INK, MID, STEEL, OURS, FONT_MONO } from './charts.js';

const W = 1080;
const H = 322;
const STACK_W = 660; // below this the five columns cannot hold their labels, so we stack vertically

const N_LAYERS = 28;
const TAP = 20; // layers 1..20 are kept; 21..28 are never computed
const SPINE_Y = 150;
const PITCH = 6; // vertical distance between two layer bars

const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

function header(x, y, n, label) {
  const g = svgEl('g');
  g.appendChild(svgText(x, y, String(n), { fill: OURS, 'font-size': 11, 'font-weight': 700 }));
  g.appendChild(svgText(x + 13, y, label, { fill: MID, 'font-size': 11 }));
  return g;
}

// keyW: where the value column starts, measured from the chip's left edge. Fixed, not derived from
// the label, so every value in a column lines up.
//
// Returns its parts, because the highlight is a colour change and not an opacity change. Staging
// this figure by fading stages between 0.42 and 1 made every stage look half-off and none of them
// look lit; at rest everything now sits at full strength in neutral ink, and the stage the run is
// currently in turns green. That is a change you can see across the room.
function chip(x, y, w, h, label, value, { keyW = 42 } = {}) {
  const g = svgEl('g');
  const rect = svgEl('rect', {
    x, y, width: w, height: h, rx: 4, fill: 'rgba(41,40,39,0.035)', stroke: STEEL, 'stroke-width': 1,
  });
  g.appendChild(rect);
  const ty = y + h / 2 + 4;
  const key = svgText(x + 10, ty, label, { fill: INK, 'font-size': 11, 'font-weight': 700 });
  g.appendChild(key);
  if (value) g.appendChild(svgText(x + keyW, ty, value, { fill: MID, 'font-size': 11 }));
  return { g, rect, key };
}

const LIT = { stroke: OURS, fill: 'rgba(47,93,80,0.12)', strokeWidth: 1.7 };
const DIM = { stroke: STEEL, fill: 'rgba(41,40,39,0.035)', strokeWidth: 1 };

// A connector that draws itself. The head is hidden until the line reaches it — an arrowhead
// floating on its own, with no shaft behind it, was the most obviously wrong thing in the figure.
function connector(x1, x2, y) {
  const g = svgEl('g');
  const len = x2 - 6 - x1;
  const line = svgEl('path', { d: `M${x1} ${y} L${x2 - 6} ${y}`, stroke: STEEL, 'stroke-width': 1.5, fill: 'none' });
  const head = svgEl('path', { d: `M${x2 - 7} ${y - 3.6} L${x2} ${y} L${x2 - 7} ${y + 3.6}`, fill: STEEL, opacity: 1 });
  g.appendChild(line); g.appendChild(head);
  return { g, line, head, len };
}

export function mountDecisionFlow(container) {
  if (!container) return;
  let tl = null;
  let ants = null;
  let io = null;

  const draw = () => {
    if (tl) { tl.kill(); tl = null; }
    if (ants) { ants.kill(); ants = null; }
    if (io) { io.disconnect(); io = null; }
    container.innerHTML = '';
    const w = fitWidth(container, W);
    const stacked = w < STACK_W;
    const h = stacked ? 560 : H;
    container.style.background = '#f0eeeb';
    container.style.maxWidth = `${W}px`;
    const svg = svgEl('svg', { viewBox: `0 0 ${stacked ? STACK_W : W} ${h}`, width: '100%', role: 'img' });
    const title = svgEl('title');
    title.textContent = 'How one decision is computed: the request tuple enters a truncated backbone, two hidden states are read off the tap layer, a bilinear form scores each candidate, and a gate outside the softmax decides whether to answer at all.';
    svg.appendChild(title);
    container.appendChild(svg);

    // Narrow screens get the same five groups laid out downward, and no animation. A flow you have
    // to scroll to follow is not a flow.
    if (stacked) {
      [['the request', 'S, Q, C and a type — candidates are text, supplied per call'],
       ['one pass', `Qwen3 trunk, cut at layer ${TAP} of ${N_LAYERS}`],
       ['two states', 'h_D for the decision, h_j for each candidate'],
       ['a score', 's_j = uᵀA v_j + g(u, v_j), per candidate'],
       ['one distribution', 'a gate r outside the softmax, plus p∅']].forEach(([name, sub], i) => {
        const y = 40 + i * 100;
        svg.appendChild(header(24, y, i + 1, name));
        svg.appendChild(svgEl('rect', { x: 24, y: y + 14, width: STACK_W - 48, height: 52, rx: 5, fill: 'rgba(41,40,39,0.04)', stroke: STEEL }));
        svg.appendChild(svgText(36, y + 45, sub, { fill: INK, 'font-size': 12 }));
      });
      registerChart(container, draw);
      return;
    }

    const COL = { req: 0, stack: 250, states: 470, score: 672, out: 872 };
    const HEAD_Y = 40;
    const layerY = (i) => SPINE_Y + (TAP - i) * PITCH; // layer TAP sits exactly on the spine

    // ---- 1. the request -------------------------------------------------------------------
    svg.appendChild(header(COL.req, HEAD_Y, 1, 'the request'));
    const chips = [['S', 'ticket #7734'], ['Q', 'which team owns this?'], ['C', '{billing, refunds, …}'], ['type', 'choice']]
      .map(([k, v], i) => {
        const c = chip(COL.req, 85 + i * 34, 196, 28, k, v, { keyW: 46 });
        svg.appendChild(c.g);
        return c;
      });
    svg.appendChild(svgText(COL.req, 288, 'C is text, supplied per call —', { fill: MID, 'font-size': 10 }));
    svg.appendChild(svgText(COL.req, 302, 'there is no output index for a label', { fill: MID, 'font-size': 10 }));
    const c1 = connector(206, COL.stack - 6, SPINE_Y);
    svg.appendChild(c1.g);

    // ---- 2. the pass, cut at the tap ------------------------------------------------------
    svg.appendChild(header(COL.stack, HEAD_Y, 2, 'one pass'));
    // The pass used to be twenty bars each fading in on its own stagger, which read as a string of
    // lights blinking rather than as one computation moving. It is now a single mask climbing a
    // green copy of the kept layers: one continuous motion, the way a column fills.
    const clipId = `dfclip-${Math.random().toString(36).slice(2, 8)}`;
    const defs = svgEl('defs');
    const clip = svgEl('clipPath', { id: clipId });
    const stackBottom = layerY(1) + 4.5;
    // Drawn climbed. Everything in this figure renders in its finished state so that a paused
    // timeline, a reduced-motion reader and a screenshot all get a correct picture; the timeline
    // empties it at the start of each cycle and fills it again. Left at height 0 the stack would
    // lose the one fact it exists to carry -- that the pass stops at 20 of 28.
    const climbRect = svgEl('rect', { x: COL.stack - 2, y: SPINE_Y, width: 124, height: stackBottom - SPINE_Y });
    clip.appendChild(climbRect);
    defs.appendChild(clip);
    svg.appendChild(defs);

    const layerRect = (i, attrs) => svgEl('rect', {
      x: COL.stack, y: layerY(i), width: 120, height: 4.5, rx: 1.5, ...attrs,
    });
    for (let i = N_LAYERS; i >= 1; i--) {
      svg.appendChild(i <= TAP
        ? layerRect(i, { fill: INK, opacity: 0.16 })
        : layerRect(i, { fill: 'none', stroke: STEEL, 'stroke-width': 1, 'stroke-dasharray': '2 2', opacity: 0.6 }));
    }
    const climb = svgEl('g', { 'clip-path': `url(#${clipId})` });
    for (let i = TAP; i >= 1; i--) climb.appendChild(layerRect(i, { fill: OURS, opacity: 0.92 }));
    svg.appendChild(climb);
    svg.appendChild(svgText(COL.stack, 288, `layers ${TAP + 1}–${N_LAYERS} are never`, { fill: MID, 'font-size': 10 }));
    svg.appendChild(svgText(COL.stack, 302, 'computed at all', { fill: MID, 'font-size': 10 }));

    // the tap: the spine leaving layer 20. Marches continuously — it is the one line that is
    // always carrying something, because the prefix cache is reused by every later question.
    const tapLine = svgEl('path', {
      d: `M${COL.stack} ${SPINE_Y} L${COL.states - 8} ${SPINE_Y}`,
      stroke: OURS, 'stroke-width': 1.5, fill: 'none', 'stroke-dasharray': '5 4',
    });
    svg.appendChild(tapLine);
    // The label belongs in the gutter on the line it names, not on top of the stack. Printed over
    // the bars it read as a layer, which is the one thing it is not.
    svg.appendChild(svgText(COL.stack + 128, SPINE_Y - 8, `tap · layer ${TAP}`, { fill: OURS, 'font-size': 9, 'font-weight': 700 }));

    // ---- 3. the two states ----------------------------------------------------------------
    svg.appendChild(header(COL.states, HEAD_Y, 3, 'two states'));
    const forkTop = SPINE_Y - 22, forkBot = SPINE_Y + 22;
    const fork = svgEl('path', {
      d: `M${COL.states - 8} ${SPINE_Y} L${COL.states - 8} ${forkTop} L${COL.states} ${forkTop}` +
         ` M${COL.states - 8} ${SPINE_Y} L${COL.states - 8} ${forkBot} L${COL.states} ${forkBot}`,
      stroke: OURS, 'stroke-width': 1.3, fill: 'none', opacity: 0.6,
    });
    svg.appendChild(fork);
    const hD = chip(COL.states, forkTop - 14, 150, 28, 'h_D', 'decision', { keyW: 46 });
    const hJ = chip(COL.states, forkBot - 14, 150, 28, 'h_j', 'candidate', { keyW: 46 });
    svg.appendChild(hD.g); svg.appendChild(hJ.g);
    svg.appendChild(svgText(COL.states, 288, 'h_j is the mean hidden state', { fill: MID, 'font-size': 10 }));
    svg.appendChild(svgText(COL.states, 302, 'over that option’s own tokens', { fill: MID, 'font-size': 10 }));
    const c2 = connector(COL.states + 156, COL.score - 6, SPINE_Y);
    svg.appendChild(c2.g);

    // ---- 4. the score ---------------------------------------------------------------------
    svg.appendChild(header(COL.score, HEAD_Y, 4, 'a score'));
    const scoreG = svgEl('g');
    const scoreRect = svgEl('rect', { x: COL.score, y: SPINE_Y - 35, width: 156, height: 70, rx: 5, fill: DIM.fill, stroke: STEEL, 'stroke-width': 1 });
    scoreG.appendChild(scoreRect);
    const scoreKey = svgText(COL.score + 14, SPINE_Y - 12, 'sⱼ = uᵀA vⱼ', { fill: INK, 'font-size': 13, 'font-weight': 700 });
    scoreG.appendChild(scoreKey);
    scoreG.appendChild(svgText(COL.score + 14, SPINE_Y + 6, '+ g(u, vⱼ)', { fill: INK, 'font-size': 13, 'font-weight': 700 }));
    scoreG.appendChild(svgText(COL.score + 14, SPINE_Y + 24, 'bilinear + 2-layer MLP', { fill: MID, 'font-size': 9 }));
    svg.appendChild(scoreG);
    const scoreBox = { g: scoreG, rect: scoreRect, key: scoreKey };
    svg.appendChild(svgText(COL.score, 288, 'the whole set is scored', { fill: MID, 'font-size': 10 }));
    svg.appendChild(svgText(COL.score, 302, 'together, not one at a time', { fill: MID, 'font-size': 10 }));
    const c3 = connector(COL.score + 162, COL.out - 6, SPINE_Y);
    svg.appendChild(c3.g);

    // ---- 5. the distribution --------------------------------------------------------------
    svg.appendChild(header(COL.out, HEAD_Y, 5, 'one distribution'));
    const OUT_W = W - COL.out - 8;
    const gate = chip(COL.out, 84, OUT_W, 26, 'gate r', 'outside the softmax', { keyW: 60 });
    svg.appendChild(gate.g);
    // Shape only, and it says so. No probability on this site is invented.
    const BARS = [['billing', 0.62], ['refunds', 0.2], ['shipping', 0.08], ['p∅', 0.1]];
    const barX = COL.out + 62;
    const barMax = W - 8 - barX;
    const bars = BARS.map(([label, frac], i) => {
      const y = 128 + i * 26;
      svg.appendChild(svgText(COL.out, y + 9, label, { fill: label === 'p∅' ? OURS : INK, 'font-size': 10 }));
      svg.appendChild(svgEl('rect', { x: barX, y, width: barMax, height: 10, rx: 2, fill: 'rgba(41,40,39,0.05)' }));
      const fill = svgEl('rect', {
        x: barX, y, width: barMax * frac, height: 10, rx: 2,
        fill: label === 'p∅' ? OURS : INK, opacity: label === 'p∅' ? 0.75 : 0.5,
      });
      svg.appendChild(fill);
      return fill;
    });
    svg.appendChild(svgText(COL.out, 288, 'shape only, not measurements —', { fill: MID, 'font-size': 10 }));
    svg.appendChild(svgText(COL.out, 302, 'they sum to one, p∅ included', { fill: MID, 'font-size': 10 }));

    registerChart(container, draw);

    // ---- motion ----------------------------------------------------------------------------
    const g = window.gsap;
    if (!g || reduced()) return; // what is drawn above is already the finished state

    const conns = [c1, c2, c3];
    const boxes = [...chips, hD, hJ, scoreBox, gate];
    const stackBottomY = layerY(1) + 4.5;
    const climbH = stackBottomY - SPINE_Y;

    // Strong ease-out for anything arriving; ease-in-out for the one thing that travels across the
    // figure. Only the marching tap dashes are linear, because constant motion should be constant.
    const ARRIVE = 'expo.out';
    const TRAVEL = 'power3.inOut';

    // light(target, at) -- the stage turns green. dark(...) puts it back.
    const light = (c, at) => {
      tl.to(c.rect, { stroke: LIT.stroke, fill: LIT.fill, strokeWidth: LIT.strokeWidth, duration: 0.45 }, at)
        .to(c.key, { fill: OURS, duration: 0.45 }, at);
    };

    // Explanatory, not UI: the sub-300ms rule is for things a user triggers and waits on. Each beat
    // here is still under 900ms; the run takes ~10s, holds on the finished picture, resets, and
    // waits before going again.
    tl = g.timeline({ paused: true, repeat: -1, repeatDelay: 1.4, defaults: { ease: ARRIVE } });

    tl.set(boxes.map((c) => c.rect), { stroke: DIM.stroke, fill: DIM.fill, strokeWidth: DIM.strokeWidth })
      .set(boxes.map((c) => c.key), { fill: INK })
      .set(fork, { opacity: 0.2 })
      .set(climbRect, { attr: { y: stackBottomY, height: 0 } })
      .set(bars, { scaleX: 0, transformOrigin: 'left center' })
      .set(conns.map((c) => c.head), { opacity: 0, fill: STEEL })
      .set(conns.map((c) => c.line), { stroke: STEEL, 'stroke-dasharray': (i) => `${conns[i].len}`, 'stroke-dashoffset': (i) => conns[i].len });

    // The tap is always carrying something -- the prefix cache is reused by every later question --
    // so its dashes march forever. That tween must NOT live on the timeline: a child with
    // repeat: -1 gives its parent an infinite duration, so the parent's own repeat/repeatDelay can
    // never fire. That is exactly why this figure ran once and stopped.
    ants = g.to(tapLine, { 'stroke-dashoffset': -36, duration: 2.2, repeat: -1, ease: 'none', paused: true });

    // 1. the request lights row by row
    chips.forEach((c, i) => light(c, 0.3 + i * 0.16));

    // 2. the connector draws in green, its head arrives with it, then the pass climbs the stack
    tl.to(c1.line, { stroke: OURS, 'stroke-dashoffset': 0, duration: 0.6 }, 1.25)
      .to(c1.head, { opacity: 1, fill: OURS, duration: 0.2 }, 1.7)
      .to(climbRect, { attr: { y: SPINE_Y, height: climbH }, duration: 2.0, ease: TRAVEL }, 1.9);

    // 3. out along the tap into the fork
    tl.to(fork, { opacity: 1, duration: 0.4 }, 3.9);
    light(hD, 4.05);
    light(hJ, 4.2);

    // 4. into the score
    tl.to(c2.line, { stroke: OURS, 'stroke-dashoffset': 0, duration: 0.55 }, 4.9)
      .to(c2.head, { opacity: 1, fill: OURS, duration: 0.2 }, 5.3);
    light(scoreBox, 5.45);

    // 5. the gate, then the bars grow out of it
    tl.to(c3.line, { stroke: OURS, 'stroke-dashoffset': 0, duration: 0.55 }, 6.1)
      .to(c3.head, { opacity: 1, fill: OURS, duration: 0.2 }, 6.5);
    light(gate, 6.65);
    tl.to(bars, { scaleX: 1, duration: 0.6, stagger: 0.09 }, 7.1);

    // 6. hold on the finished picture, then reset in one soft pass. A loop that visibly rewinds
    //    reads as a bug, so the reset is a fade rather than a reversal, and it is faster than the
    //    run that earned it. repeatDelay above gives the blank beat before it starts again.
    const RESET = 8.9;
    tl.to({}, { duration: 1.0 }, 7.9)
      .to(boxes.map((c) => c.rect), { stroke: DIM.stroke, fill: DIM.fill, strokeWidth: DIM.strokeWidth, duration: 0.55, ease: 'power2.inOut' }, RESET)
      .to(boxes.map((c) => c.key), { fill: INK, duration: 0.55, ease: 'power2.inOut' }, RESET)
      .to(fork, { opacity: 0.2, duration: 0.55 }, RESET)
      .to(climbRect, { attr: { y: stackBottomY, height: 0 }, duration: 0.6, ease: 'power2.inOut' }, RESET)
      .to(bars, { scaleX: 0, duration: 0.5, ease: 'power2.inOut' }, RESET)
      .to(conns.map((c) => c.head), { opacity: 0, duration: 0.3 }, RESET)
      .to(conns.map((c) => c.line), { stroke: STEEL, 'stroke-dashoffset': (i) => conns[i].len, duration: 0.55 }, RESET);

    io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { tl.play(); ants.play(); } else { tl.pause(); ants.pause(); } });
    }, { threshold: 0.2 });
    io.observe(container);
  };

  draw();
}
