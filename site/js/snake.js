// Pure Snake game engine. No DOM, no rendering framework - state in, state out.
// Canonical direction order (content-spec.md R7): up, down, left, right - used for
// safeMoves(), describe()'s unsafe-move and "Safe moves:" lists, and candidate order
// wherever a caller offers moves to the model.
const MOVES = ['up', 'down', 'left', 'right'];
const DELTA = { up: { x: 0, y: -1 }, down: { x: 0, y: 1 }, left: { x: -1, y: 0 }, right: { x: 1, y: 0 } };
const OPPOSITE = { up: 'down', down: 'up', left: 'right', right: 'left' };

export class Snake {
  constructor({ w = 10, h = 10 } = {}) {
    this.w = w;
    this.h = h;
    this.reset();
  }

  reset() {
    const cx = Math.floor(this.w / 2);
    const cy = Math.floor(this.h / 2);
    this.body = [{ x: cx, y: cy }, { x: cx - 1, y: cy }, { x: cx - 2, y: cy }];
    this.dir = 'right';
    this.score = 0;
    this.steps = 0;
    this.dead = false;
    this.food = this._placeFood();
    return this.state();
  }

  get head() {
    return this.body[0];
  }

  _placeFood() {
    let p;
    do {
      p = { x: Math.floor(Math.random() * this.w), y: Math.floor(Math.random() * this.h) };
    } while (this.body.some((b) => b.x === p.x && b.y === p.y));
    return p;
  }

  _hits(x, y, bodyToCheck) {
    if (x < 0 || x >= this.w || y < 0 || y >= this.h) return true;
    return bodyToCheck.some((b) => b.x === x && b.y === y);
  }

  // A move into the current tail cell is safe (the tail vacates that turn)
  // unless the move also eats food, in which case the tail stays put.
  _targetBody(move) {
    const d = DELTA[move];
    const nx = this.head.x + d.x;
    const ny = this.head.y + d.y;
    const willEat = nx === this.food.x && ny === this.food.y;
    return { nx, ny, willEat, bodyToCheck: willEat ? this.body : this.body.slice(0, -1) };
  }

  safeMoves() {
    return MOVES.filter((m) => {
      if (OPPOSITE[this.dir] === m && this.body.length > 1) return false;
      const { nx, ny, bodyToCheck } = this._targetBody(m);
      return !this._hits(nx, ny, bodyToCheck);
    });
  }

  // Copies head/body/food (like Doom/Drive's state()) rather than handing out live references
  // - callers (replay recorders, renderers keeping a prev/curr pair for interpolation) need a
  // frozen snapshot; `this.body` keeps getting unshift/pop'd in place after this returns.
  state() {
    return {
      head: { ...this.head },
      dir: this.dir,
      body: this.body.map((b) => ({ ...b })),
      food: { ...this.food },
      w: this.w,
      h: this.h,
      score: this.score,
      steps: this.steps,
      dead: this.dead,
    };
  }

  step(move) {
    if (this.dead) return this.state();
    if (!DELTA[move]) move = this.dir;
    if (OPPOSITE[this.dir] === move && this.body.length > 1) move = this.dir; // ignore illegal reversal
    const { nx, ny, willEat, bodyToCheck } = this._targetBody(move);
    if (this._hits(nx, ny, bodyToCheck)) {
      this.dead = true;
      return this.state();
    }
    this.dir = move;
    this.body.unshift({ x: nx, y: ny });
    if (willEat) {
      this.score += 1;
      this.food = this._placeFood();
    } else {
      this.body.pop();
    }
    this.steps += 1;
    return this.state();
  }

  render() {
    const grid = Array.from({ length: this.h }, () => Array(this.w).fill('·'));
    this.body.forEach((b, i) => {
      grid[b.y][b.x] = i === 0 ? '▓' : '█';
    });
    grid[this.food.y][this.food.x] = '◆';
    const top = '┌' + '─'.repeat(this.w) + '┐';
    const bottom = '└' + '─'.repeat(this.w) + '┘';
    const rows = grid.map((row) => '│' + row.join('') + '│');
    return [top, ...rows, bottom].join('\n');
  }

  // Which single-step moves close the food gap: [larger-axis move, smaller-axis move]; a
  // missing axis is null, |dx| == |dy| counts the horizontal axis as the larger one.
  gapMoves() {
    const dx = this.food.x - this.head.x;
    const dy = this.food.y - this.head.y;
    const hx = dx > 0 ? 'right' : dx < 0 ? 'left' : null;
    const vy = dy > 0 ? 'down' : dy < 0 ? 'up' : null;
    return Math.abs(dx) >= Math.abs(dy) ? [hx, vy] : [vy, hx];
  }

  // The model's candidates = the safe moves (the legality guardrail), canonical order.
  candidates() {
    return this.safeMoves();
  }

  // Demo-spec v4 grammar: only the situations that apply, as agent-subject sentences that
  // pre-compute the comparison (which axis is the larger gap), in this fixed order:
  //   [The snake must move {A} to close the {larger gap|gap} to the food.]   A safe
  //   [The snake can also move {B} to close the smaller gap.]                B exists and safe
  //   [Moving {d} hits the wall. | Moving {d} hits the snake's body.]        per unsafe d, canonical
  // "must" > "can also" is the precedence; the probe (n=62) reads it at .984 with the plain
  // QUESTION, while every variant that spelled the rules out in the question scored .82-.95 with
  // p_null ~.9 (four direction rules and no default are off the trained 3-rules+default shape).
  describe() {
    const safe = new Set(this.safeMoves());
    const [a, b] = this.gapMoves();
    const sentences = [];
    if (a && safe.has(a)) sentences.push(`The snake must move ${a} to close the ${b ? 'larger gap' : 'gap'} to the food.`);
    if (b && safe.has(b)) sentences.push(`The snake can also move ${b} to close the smaller gap.`);
    for (const m of MOVES) {
      if (safe.has(m)) continue;
      const { nx, ny } = this._targetBody(m);
      const wall = nx < 0 || nx >= this.w || ny < 0 || ny >= this.h;
      sentences.push(`Moving ${m} hits ${wall ? 'the wall' : "the snake's body"}.`);
    }
    return sentences.join(' ');
  }
}

// The policy the state sentences encode, in precedence order (documentation + greedyPolicy).
// Following it literally eats ~28 food in 300 ticks on a 10x10 board (seed 7 simulation).
export const RULES = {
  'close the larger gap': 'move along the axis with the larger distance to the food when that move is safe',
  'close the smaller gap': 'else move along the other axis when that move is safe',
  'any safe move': 'else the first safe move in up, down, left, right order',
};
// Sent verbatim as the query (presets.json snake.question). No rule list: see describe().
export const QUESTION = 'Which move brings the snake closer to the food without dying?';

// Scripted policy = RULES applied literally; the gold the model is measured against.
export function greedyPolicy(engine) {
  const safe = engine.safeMoves();
  const [a, b] = engine.gapMoves();
  if (a && safe.includes(a)) return a;
  if (b && safe.includes(b)) return b;
  return safe[0] ?? null;
}

function selfTest() {
  const s = new Snake({ w: 6, h: 6 });
  console.assert(s.w === 6 && s.h === 6, 'constructor honors explicit w/h');
  console.assert(new Snake().w === 10 && new Snake().h === 10, 'default board is 10x10');
  console.assert(s.body.length === 3, 'starts with body length 3');
  console.assert(s.dir === 'right', 'starts moving right');
  const moves = s.safeMoves();
  console.assert(Array.isArray(moves) && moves.length > 0, 'has safe moves from start');
  console.assert(!moves.includes('left'), 'cannot reverse into own neck');
  console.assert(JSON.stringify(moves) === JSON.stringify(moves.slice().sort((a, b) => ['up', 'down', 'left', 'right'].indexOf(a) - ['up', 'down', 'left', 'right'].indexOf(b))), 'safeMoves is in canonical order');
  const before = s.steps;
  s.step(moves[0]);
  console.assert(s.steps === before + 1, 'step increments steps');
  console.assert(s.render().split('\n').length === s.h + 2, 'render has h+2 rows (border)');
  console.assert(s.render().startsWith('┌') && s.render().includes('┘'), 'render has box-drawing border');

  // exact describe() grammar (demo-spec v4)
  const s2 = new Snake({ w: 10, h: 10 });
  s2.body = [{ x: 4, y: 4 }, { x: 3, y: 4 }, { x: 2, y: 4 }];
  s2.dir = 'right';
  s2.food = { x: 7, y: 2 }; // 3 cells right, 2 cells up of head (4,4)
  console.assert(
    s2.describe() === "The snake must move right to close the larger gap to the food. The snake can also move up to close the smaller gap. Moving left hits the snake's body.",
    'describe: must (larger gap), can also (smaller gap), reversal counts as the body'
  );
  console.assert(greedyPolicy(s2) === 'right' && s2.candidates().join() === 'up,down,right', 'greedy takes the larger gap; candidates = safe moves');

  // single axis: "the gap", no smaller-gap sentence
  const s3 = new Snake({ w: 10, h: 10 });
  s3.body = [{ x: 4, y: 4 }, { x: 3, y: 4 }, { x: 2, y: 4 }];
  s3.dir = 'right';
  s3.food = { x: 7, y: 4 };
  console.assert(s3.describe().startsWith('The snake must move right to close the gap to the food. Moving left'), 'single-axis food says "the gap"');

  // larger-gap move unsafe: only the smaller-gap sentence + the wall fact; greedy falls through
  const s5 = new Snake({ w: 10, h: 10 });
  s5.body = [{ x: 9, y: 4 }, { x: 8, y: 4 }, { x: 7, y: 4 }];
  s5.dir = 'right';
  s5.food = { x: 6, y: 9 }; // 3 left, 5 down -> A = down, B = left
  console.assert(greedyPolicy(s5) === 'down', 'larger gap wins');
  s5.food = { x: 8, y: 9 }; // 1 left, 5 down -> A = down (safe), B = left (reversal, unsafe)
  console.assert(s5.describe() === "The snake must move down to close the larger gap to the food. Moving left hits the snake's body. Moving right hits the wall.", 'unsafe smaller-gap move is a hit sentence, not a can-also');
  s5.body = [{ x: 9, y: 9 }, { x: 8, y: 9 }, { x: 7, y: 9 }];
  s5.food = { x: 8, y: 5 }; // 1 left, 4 up -> A = up
  s5.dir = 'right';
  console.assert(greedyPolicy(s5) === 'up', 'up closes the larger gap');
  s5.body = [{ x: 9, y: 9 }, { x: 9, y: 8 }, { x: 9, y: 7 }];
  s5.dir = 'down';
  s5.food = { x: 5, y: 9 }; // 4 left, 0 -> A = left, safe
  console.assert(s5.describe() === 'The snake must move left to close the gap to the food. Moving up hits the snake\'s body. Moving down hits the wall. Moving right hits the wall.', 'three hit sentences in canonical order');
  s5.food = { x: 9, y: 2 }; // 0, 7 up -> A = up (unsafe: neck), no B -> default = first safe = left
  console.assert(s5.describe() === "Moving up hits the snake's body. Moving down hits the wall. Moving right hits the wall." && greedyPolicy(s5) === 'left', 'nothing closes the gap safely -> only hit sentences, greedy = first safe move');
  console.assert(QUESTION === 'Which move brings the snake closer to the food without dying?' && Object.keys(RULES).length === 3, 'QUESTION/RULES exported');

  // drive into a wall deliberately to confirm death + unsafe-move reporting, canonical order
  const s4 = new Snake({ w: 6, h: 6 });
  s4.body = [{ x: 5, y: 0 }, { x: 4, y: 0 }, { x: 3, y: 0 }];
  s4.dir = 'right';
  s4.food = { x: 0, y: 5 };
  console.assert(!s4.safeMoves().includes('right'), 'wall move excluded from safeMoves');
  console.assert(s4.describe().includes('Moving right hits the wall.'), 'describe lists the unsafe move with the exact phrase');
  s4.step('right');
  console.assert(s4.dead === true, 'stepping into a wall kills the snake');

  console.log('snake.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
