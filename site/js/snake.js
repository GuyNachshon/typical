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

  // Exact template (content-spec.md R7), 1-based column/row, row 1 = top:
  // "Snake game on a {w} by {h} board. The head is at column {hx}, row {hy} (row 1 is
  // the top). The food is {|dx|} cells to the {left|right} and {|dy|} cells {up|down}.
  // [Moving {d} collides with a wall or the body. - one per unsafe direction, canonical
  // order] Safe moves: {comma list}."
  // {distanceFacts:true} (pre-registered fallback (b), content-spec.md R7) inserts one
  // sentence per SAFE move - "Moving {d} reduces/increases/does not change the distance
  // to the food." (Manhattan) - after the collision sentences, before "Safe moves:".
  // Default (no options) is the plain template, unchanged.
  describe({ distanceFacts = false } = {}) {
    const dx = this.food.x - this.head.x;
    const dy = this.food.y - this.head.y;
    const sentences = [
      `Snake game on a ${this.w} by ${this.h} board.`,
      `The head is at column ${this.head.x + 1}, row ${this.head.y + 1} (row 1 is the top).`,
    ];
    const xDesc = dx === 0 ? '' : `${Math.abs(dx)} cells to the ${dx > 0 ? 'right' : 'left'}`;
    const yDesc = dy === 0 ? '' : `${Math.abs(dy)} cells ${dy > 0 ? 'down' : 'up'}`;
    const foodDesc = [xDesc, yDesc].filter(Boolean).join(' and ');
    if (foodDesc) sentences.push(`The food is ${foodDesc}.`);
    const safeMoves = this.safeMoves();
    const safe = new Set(safeMoves);
    MOVES.filter((m) => !safe.has(m)).forEach((m) => {
      sentences.push(`Moving ${m} collides with a wall or the body.`);
    });
    if (distanceFacts) {
      const distBefore = Math.abs(dx) + Math.abs(dy);
      MOVES.filter((m) => safe.has(m)).forEach((m) => {
        const d = DELTA[m];
        const distAfter = Math.abs(this.food.x - (this.head.x + d.x)) + Math.abs(this.food.y - (this.head.y + d.y));
        const verb = distAfter < distBefore ? 'reduces' : distAfter > distBefore ? 'increases' : 'does not change';
        sentences.push(`Moving ${m} ${verb} the distance to the food.`);
      });
    }
    sentences.push(`Safe moves: ${MOVES.filter((m) => safe.has(m)).join(', ')}.`);
    return sentences.join(' ');
  }
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

  // exact describe() template, 1-based column/row
  const s2 = new Snake({ w: 10, h: 10 });
  s2.body = [{ x: 4, y: 4 }, { x: 3, y: 4 }, { x: 2, y: 4 }];
  s2.dir = 'right';
  s2.food = { x: 7, y: 2 }; // 3 cells right, 2 cells up of head (4,4)
  const desc = s2.describe();
  console.assert(
    desc.startsWith('Snake game on a 10 by 10 board. The head is at column 5, row 5 (row 1 is the top). The food is 3 cells to the right and 2 cells up.'),
    'describe matches the exact spec template'
  );
  console.assert(/Safe moves: [a-z]+(, [a-z]+)*\.$/.test(desc), 'describe ends with a Safe moves list');
  console.assert(!desc.includes('reduces the distance'), 'default describe() omits distance facts');

  // fallback (b): distanceFacts option, one sentence per safe move, before "Safe moves:"
  const factsDesc = s2.describe({ distanceFacts: true });
  console.assert(
    factsDesc.includes('Moving up reduces the distance to the food. Moving down increases the distance to the food. Moving right reduces the distance to the food. Safe moves:'),
    'distanceFacts inserts one reduces/increases sentence per safe move, canonical order, before Safe moves:'
  );

  // axis omission: food directly to the right - no "up/down" clause
  const s3 = new Snake({ w: 10, h: 10 });
  s3.body = [{ x: 4, y: 4 }, { x: 3, y: 4 }, { x: 2, y: 4 }];
  s3.dir = 'right';
  s3.food = { x: 7, y: 4 };
  console.assert(s3.describe().includes('The food is 3 cells to the right.'), 'omits the up/down clause when dy is 0');

  // drive into a wall deliberately to confirm death + unsafe-move reporting, canonical order
  const s4 = new Snake({ w: 6, h: 6 });
  s4.body = [{ x: 5, y: 0 }, { x: 4, y: 0 }, { x: 3, y: 0 }];
  s4.dir = 'right';
  s4.food = { x: 0, y: 5 };
  console.assert(!s4.safeMoves().includes('right'), 'wall move excluded from safeMoves');
  console.assert(s4.describe().includes('Moving right collides with a wall or the body.'), 'describe lists the unsafe move with the exact phrase');
  s4.step('right');
  console.assert(s4.dead === true, 'stepping into a wall kills the snake');

  console.log('snake.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
