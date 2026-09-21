// ASCII arena FPS engine. Pure state machine, no DOM, no deps - matches site/js/snake.js's
// shape: state()/describe()/render()/candidates()/step()/selfTest(). The raycast render is a
// simple ray-marcher (fixed small steps, not full DDA) - the map is 24x16, so marching is cheap
// and much easier to get right than a DDA/grid-traversal implementation.

// Demo-spec v3 grammar (30/30 on the probe set, chance .33; the no-rules control is .60): the
// model decides an intent from two facts (badly hurt? enemy in sight?) and the engine aims -
// resolve() turns the intent into the key press. Letting the model aim does not work: with
// shoot / turn left / turn right as candidates it answers "shoot" for any enemy mention
// (turns 0/12 over five phrasings) - the text head cannot read a bearing. RULES order =
// precedence order = candidate order.
export const RULES = {
  retreat: 'applies when the player is badly hurt',
  engage: 'applies when the player has an enemy in sight',
  explore: 'applies when the player sees no enemy',
};
export const INTENTS = Object.keys(RULES);
// Rendered exactly like inference/typical/core.py::query_text; sent verbatim as the query by
// render-doom.js, render-realdoom.js and the recorders.
export const QUESTION =
  'Which action applies for the player? Apply the rules in the stated precedence order; the first matching rule wins.\n' +
  Object.entries(RULES).map(([k, v]) => `${k}: ${v}`).join('  ');
export const HURT_BELOW = 30; // health - "the player is badly hurt"

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Fixed 24x16 arena. '#' and 'O' (pillar) are solid; 'D' (door) is walkable floor with a
// distinct minimap glyph; 'a'/'h' mark initial ammo/health pickups (read once in reset(),
// then the tile becomes floor - pickups live in `this.items`, not the grid).
const RAW_MAP = [
  '########################',
  '#......................#',
  '#.........#............#',
  '#.........#...a........#',
  '#.........#.....OO.....#',
  '#.........D............#',
  '#.........#............#',
  '#.........#............#',
  '#.........#...##D###...#',
  '#.........#..........a.#',
  '#....h....D............#',
  '#.........#.....h......#',
  '#.....O...#............#',
  '#.........#............#',
  '#......................#',
  '########################',
];
const W = RAW_MAP[0].length;
const H = RAW_MAP.length;
const SOLID = new Set(['#', 'O']);

// 8-way headings, clockwise from North. turn left/right move the index by -1/+1 mod 8.
const HEADINGS = [
  { name: 'N', dx: 0, dy: -1 },
  { name: 'NE', dx: 1, dy: -1 },
  { name: 'E', dx: 1, dy: 0 },
  { name: 'SE', dx: 1, dy: 1 },
  { name: 'S', dx: 0, dy: 1 },
  { name: 'SW', dx: -1, dy: 1 },
  { name: 'W', dx: -1, dy: 0 },
  { name: 'NW', dx: -1, dy: -1 },
];
// Minimap player glyph: 4-way arrow, nearest cardinal for diagonal headings.
const ARROW = ['▲', '▶', '▶', '▼', '▼', '▼', '◀', '◀'];

const ACTIONS = ['move forward', 'move back', 'turn left', 'turn right', 'shoot'];
const AWARE_RANGE = 8; // describe()/enemy-aggro radius, in cells
const SHOOT_RANGE = 6;
const DAMAGE = 10;
const FOV = Math.PI / 3;
const COLS = 72;
const ROWS = 16;
const MAX_CAST = 16;
const WALL_SHADE = ['█', '▓', '▒', '░', '·']; // near -> far

function chebyshev(x1, y1, x2, y2) {
  return Math.max(Math.abs(x1 - x2), Math.abs(y1 - y2));
}

// Bearing of (ex,ey) relative to a viewer at (px,py) facing heading index h, bucketed into
// the four quadrants the grammar uses (see describe()).
function bearingOf(px, py, h, ex, ey) {
  const { dx, dy } = HEADINGS[h];
  const vx = ex - px;
  const vy = ey - py;
  const fwd = dx * vx + dy * vy;
  const right = -dy * vx + dx * vy;
  const angle = (Math.atan2(right, fwd) * 180) / Math.PI;
  if (angle >= -45 && angle <= 45) return 'ahead';
  if (angle > 45 && angle < 135) return 'right';
  if (angle <= -45 && angle > -135) return 'left';
  return 'behind';
}

// True iff (ex,ey) sits exactly on the 8-way ray fired from (px,py) along heading h - the
// same cells shoot() walks - so "ahead in the crosshair" means "shoot would hit this cell".
function onRay(px, py, h, ex, ey) {
  const { dx, dy } = HEADINGS[h];
  const vx = ex - px;
  const vy = ey - py;
  if (vx === 0 && vy === 0) return false;
  if (dx === 0) return vx === 0 && Math.sign(vy) === dy;
  if (dy === 0) return vy === 0 && Math.sign(vx) === dx;
  return vx * dy === vy * dx && Math.sign(vx) === dx && Math.sign(vy) === dy;
}

export class Doom {
  constructor({ seed = 1 } = {}) {
    this.rng = mulberry32(seed);
    this.reset();
  }

  reset() {
    this.grid = RAW_MAP.map((r) => r.split(''));
    this.items = [];
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const c = this.grid[y][x];
        if (c === 'a' || c === 'h') {
          this.items.push({ x, y, type: c === 'a' ? 'ammo' : 'health', taken: false });
          this.grid[y][x] = '.';
        }
      }
    }
    this.player = { x: 3, y: 3, h: 2, health: 100, ammo: 12 }; // h=2 -> facing E
    this.enemies = [
      { x: 20, y: 3, alive: true },
      { x: 20, y: 12, alive: true },
      { x: 3, y: 12, alive: true },
      { x: 12, y: 6, alive: true },
    ];
    this.dead = false;
    this.score = 0;
    this.ticks = 0;
    return this.state();
  }

  _solid(x, y) {
    if (x < 0 || x >= W || y < 0 || y >= H) return true;
    return SOLID.has(this.grid[y][x]);
  }

  _occupied(x, y, self) {
    if (this._solid(x, y)) return true;
    if (x === this.player.x && y === this.player.y) return true;
    return this.enemies.some((o) => o !== self && o.alive && o.x === x && o.y === y);
  }

  state() {
    return {
      player: { ...this.player },
      enemies: this.enemies.map((e) => ({ ...e })),
      items: this.items.map((i) => ({ ...i })),
      dead: this.dead,
      score: this.score,
      ticks: this.ticks,
    };
  }

  _legalActions() {
    if (this.dead) return [];
    const p = this.player;
    const { dx, dy } = HEADINGS[p.h];
    const legal = new Set(['turn left', 'turn right']);
    if (!this._solid(p.x + dx, p.y + dy)) legal.add('move forward');
    if (!this._solid(p.x - dx, p.y - dy)) legal.add('move back');
    if (p.ammo > 0) legal.add('shoot');
    return ACTIONS.filter((a) => legal.has(a));
  }

  // The model's candidates are the three intents; all are always offered while alive.
  candidates() {
    return this.dead ? [] : [...INTENTS];
  }

  // Enemies within AWARE_RANGE, crosshair (on the shoot ray) first, then nearest first.
  _visible() {
    const p = this.player;
    return this.enemies
      .filter((e) => e.alive && chebyshev(e.x, e.y, p.x, p.y) <= AWARE_RANGE)
      .map((e) => ({ e, dist: chebyshev(e.x, e.y, p.x, p.y), crosshair: onRay(p.x, p.y, p.h, e.x, e.y), bearing: bearingOf(p.x, p.y, p.h, e.x, e.y) }))
      .sort((a, b) => (a.crosshair === b.crosshair ? a.dist - b.dist : a.crosshair ? -1 : 1));
  }

  // Intent -> key press (the engine aims; the page says so). Raw actions pass through so the
  // human override keeps working.
  resolve(move) {
    if (!INTENTS.includes(move)) return move;
    const legal = this._legalActions();
    const pick = (...prefs) => prefs.find((a) => legal.includes(a)) ?? legal[0] ?? null;
    const target = this._visible()[0];
    if (move === 'retreat') return pick('move back', 'turn right');
    if (move === 'engage' && target) {
      if (target.crosshair && target.dist <= SHOOT_RANGE) return pick('shoot', 'move back');
      if (target.bearing === 'left') return pick('turn left');
      if (target.bearing === 'right' || target.bearing === 'behind') return pick('turn right');
      return pick('move forward', 'turn right');
    }
    return pick('move forward', 'turn right'); // explore, or engage with nothing in sight
  }

  safeMoves() {
    return this.candidates();
  }

  _pickup(x, y) {
    const item = this.items.find((i) => !i.taken && i.x === x && i.y === y);
    if (!item) return;
    item.taken = true;
    if (item.type === 'ammo') this.player.ammo = Math.min(20, this.player.ammo + 5);
    else this.player.health = Math.min(100, this.player.health + 25);
  }

  _shootTarget() {
    const p = this.player;
    const { dx, dy } = HEADINGS[p.h];
    let x = p.x;
    let y = p.y;
    for (let step = 1; step <= SHOOT_RANGE; step++) {
      x += dx;
      y += dy;
      if (this._solid(x, y)) return null;
      const hit = this.enemies.find((e) => e.alive && e.x === x && e.y === y);
      if (hit) return hit;
    }
    return null;
  }

  _enemyTurn() {
    const p = this.player;
    for (const e of this.enemies) {
      if (!e.alive) continue;
      const dist = chebyshev(e.x, e.y, p.x, p.y);
      if (dist <= 1) {
        p.health = Math.max(0, p.health - DAMAGE);
        continue;
      }
      if (dist > AWARE_RANGE) continue;
      const adx = Math.abs(p.x - e.x);
      const ady = Math.abs(p.y - e.y);
      const sx = Math.sign(p.x - e.x);
      const sy = Math.sign(p.y - e.y);
      let stepX = sx;
      let stepY = sy;
      if (adx !== ady) {
        if (adx > ady) stepY = 0;
        else stepX = 0;
      } else if (this.rng() < 0.5) stepY = 0;
      else stepX = 0;
      let nx = e.x + stepX;
      let ny = e.y + stepY;
      if (this._occupied(nx, ny, e)) {
        nx = e.x + sx;
        ny = e.y;
        if (this._occupied(nx, ny, e)) {
          nx = e.x;
          ny = e.y + sy;
        }
      }
      if (!this._occupied(nx, ny, e)) {
        e.x = nx;
        e.y = ny;
      }
    }
  }

  step(move) {
    if (this.dead) return this.state();
    const p = this.player;
    const action = this.resolve(move);
    this.lastAction = action;
    if (this._legalActions().includes(action)) {
      if (action === 'turn left') p.h = (p.h + 7) % 8;
      else if (action === 'turn right') p.h = (p.h + 1) % 8;
      else if (action === 'move forward' || action === 'move back') {
        const { dx, dy } = HEADINGS[p.h];
        const sign = action === 'move forward' ? 1 : -1;
        p.x += dx * sign;
        p.y += dy * sign;
        this._pickup(p.x, p.y);
      } else if (action === 'shoot') {
        p.ammo -= 1;
        const hit = this._shootTarget();
        if (hit) {
          hit.alive = false;
          this.score += 1;
        }
      }
    }
    this._enemyTurn();
    this.ticks += 1;
    if (p.health <= 0) this.dead = true;
    return this.state();
  }

  // Situations-only grammar (demo-spec v3): health/ammo, then only the situations that apply,
  // worded exactly as the rule conditions, in this fixed order.
  describe() {
    const p = this.player;
    const sentences = [`The player has ${p.health} health and ${p.ammo} ammo.`];
    if (p.health < HURT_BELOW) sentences.push('The player is badly hurt.');
    const seen = this._visible();
    const where = ({ dist, crosshair, bearing }) => `${dist} cell${dist === 1 ? '' : 's'} ${crosshair ? 'ahead in the crosshair' : phraseBearing(bearing)}`;
    if (seen.length === 1) sentences.push(`The player has an enemy in sight, ${where(seen[0])}.`);
    else if (seen.length > 1) sentences.push(`The player has ${seen.length} enemies in sight; the nearest is ${where(seen[0])}.`);
    else sentences.push('The player sees no enemy.');
    const { dx, dy } = HEADINGS[p.h];
    if (this._solid(p.x + dx, p.y + dy)) sentences.push('A wall is ahead.');
    return sentences.join(' ');
  }

  // Classic Wolfenstein-style DDA raycast: steps cell-by-cell along the ray (not a fixed-size
  // march), which is what makes the N/S-vs-E/W "side" distinction below exact rather than guessed.
  _castColumn(originX, originY, rayDirX, rayDirY) {
    let mapX = Math.floor(originX);
    let mapY = Math.floor(originY);
    const deltaDistX = rayDirX === 0 ? 1e30 : Math.abs(1 / rayDirX);
    const deltaDistY = rayDirY === 0 ? 1e30 : Math.abs(1 / rayDirY);
    let stepX;
    let stepY;
    let sideDistX;
    let sideDistY;
    if (rayDirX < 0) {
      stepX = -1;
      sideDistX = (originX - mapX) * deltaDistX;
    } else {
      stepX = 1;
      sideDistX = (mapX + 1 - originX) * deltaDistX;
    }
    if (rayDirY < 0) {
      stepY = -1;
      sideDistY = (originY - mapY) * deltaDistY;
    } else {
      stepY = 1;
      sideDistY = (mapY + 1 - originY) * deltaDistY;
    }
    let side = 0;
    for (let i = 0; i < 48; i++) {
      if (sideDistX < sideDistY) {
        sideDistX += deltaDistX;
        mapX += stepX;
        side = 0; // stepped in X -> hit a vertical (E/W-facing) wall
      } else {
        sideDistY += deltaDistY;
        mapY += stepY;
        side = 1; // stepped in Y -> hit a horizontal (N/S-facing) wall
      }
      if (this._solid(mapX, mapY)) break;
    }
    const perp = side === 0 ? sideDistX - deltaDistX : sideDistY - deltaDistY;
    return { dist: Math.min(perp, MAX_CAST), side };
  }

  // Deterministic floor dither (no rng - render() must be a pure function of state): denser
  // '.' near the bottom of the screen (closer to the viewer), sparser near the wall base.
  static _floorChar(col, row) {
    const density = row / (ROWS - 1);
    const hash = (col * 13 + row * 7) % 10;
    return hash < Math.floor(density * 10) ? '.' : ' ';
  }

  render() {
    const p = this.player;
    const { dx: fdx, dy: fdy } = HEADINGS[p.h];
    const facing = Math.atan2(fdy, fdx);
    const originX = p.x + 0.5;
    const originY = p.y + 0.5;
    const dirX = Math.cos(facing);
    const dirY = Math.sin(facing);
    const planeLen = Math.tan(FOV / 2);
    const planeX = -dirY * planeLen;
    const planeY = dirX * planeLen;

    const view = Array.from({ length: ROWS }, () => new Array(COLS).fill(' '));
    const wallDist = new Array(COLS);

    for (let col = 0; col < COLS; col++) {
      const cameraX = (2 * col) / (COLS - 1) - 1;
      const rayDirX = dirX + planeX * cameraX;
      const rayDirY = dirY + planeY * cameraX;
      const { dist, side } = this._castColumn(originX, originY, rayDirX, rayDirY);
      wallDist[col] = dist;

      let shadeIdx = Math.max(0, Math.min(WALL_SHADE.length - 1, Math.floor((dist / MAX_CAST) * WALL_SHADE.length)));
      if (side === 1) shadeIdx = Math.min(WALL_SHADE.length - 1, shadeIdx + 1); // N/S faces one step darker than E/W
      const glyph = WALL_SHADE[shadeIdx];

      const lineH = Math.max(1, Math.min(ROWS, Math.round(ROWS / (dist + 0.001))));
      const top = Math.floor((ROWS - lineH) / 2);
      const bottom = top + lineH - 1;
      for (let row = top; row <= bottom; row++) view[row][col] = glyph;
      for (let row = bottom + 1; row < ROWS; row++) view[row][col] = Doom._floorChar(col, row);
      // rows above `top` stay blank (ceiling)
    }

    const relAngle = (ex, ey) => {
      const vx = ex - originX;
      const vy = ey - originY;
      const dist = Math.hypot(vx, vy);
      let angle = Math.atan2(vy, vx) - facing;
      while (angle > Math.PI) angle -= 2 * Math.PI;
      while (angle < -Math.PI) angle += 2 * Math.PI;
      const perp = dist * Math.cos(angle); // fisheye-correct, same scale as wallDist
      const col = Math.round(((angle + FOV / 2) / FOV) * (COLS - 1));
      return { perp, col };
    };

    // items: flat single-cell ground sprites, drawn before enemies so a foreground enemy wins
    for (const item of this.items) {
      if (item.taken) continue;
      const { perp, col } = relAngle(item.x + 0.5, item.y + 0.5);
      if (col < 0 || col >= COLS || perp >= MAX_CAST || perp >= wallDist[col]) continue;
      const row = Math.min(ROWS - 1, Math.floor(ROWS / 2) + 2);
      view[row][col] = item.type === 'ammo' ? 'a' : '+';
    }

    // enemies: farthest-first so a nearer enemy overdraws a farther one sharing a column
    const sprites = this.enemies
      .filter((e) => e.alive)
      .map((e) => ({ e, ...relAngle(e.x + 0.5, e.y + 0.5) }))
      .filter((s) => s.col >= 0 && s.col < COLS && s.perp < MAX_CAST)
      .sort((a, b) => b.perp - a.perp);
    for (const { perp, col } of sprites) {
      if (perp < 3) {
        // close: a 3-column blob with rounded "feet" on the bottom row
        const h = Math.max(3, Math.min(ROWS - 2, Math.round((ROWS * 0.7) / perp)));
        const top = Math.floor((ROWS - h) / 2);
        const cols = [col - 1, col, col + 1];
        const feet = ['▟', '█', '▙'];
        cols.forEach((c, i) => {
          if (c < 0 || c >= COLS || perp >= wallDist[c]) return;
          for (let row = top; row < top + h - 1; row++) view[row][c] = '█';
          view[top + h - 1][c] = feet[i];
        });
      } else if (perp < 7) {
        // mid: a single ¥ column
        if (perp >= wallDist[col]) continue;
        const h = Math.max(1, Math.min(6, Math.round((ROWS * 0.4) / perp)));
        const top = Math.floor((ROWS - h) / 2);
        for (let row = top; row < top + h; row++) view[row][col] = '¥';
      } else {
        // far: a single dot
        if (perp >= wallDist[col]) continue;
        view[Math.floor(ROWS / 2)][col] = '·';
      }
    }

    view[Math.floor(ROWS / 2)][Math.floor(COLS / 2)] = '+'; // crosshair, always on top
    const strips = view.map((row) => row.join(''));

    const mini = this.grid.map((row, y) =>
      row
        .map((c, x) => {
          if (x === p.x && y === p.y) return ARROW[p.h];
          const enemy = this.enemies.find((e) => e.alive && e.x === x && e.y === y);
          if (enemy) return 'E';
          const item = this.items.find((i) => !i.taken && i.x === x && i.y === y);
          if (item) return item.type === 'ammo' ? 'a' : '+';
          return c;
        })
        .join('')
    );
    // Minimap is exactly H rows, same as the raycast view (ROWS === H), so it sits to the
    // right of each view row rather than stacked below it.
    return strips.map((row, i) => `${row}  ${mini[i]}`).join('\n');
  }
}

function phraseBearing(b) {
  if (b === 'ahead') return 'ahead';
  if (b === 'behind') return 'behind';
  return `to the ${b}`;
}

// Scripted policy = RULES applied literally (first matching rule wins) -> an intent; step()
// resolves it to a key press. This is the gold the model is measured against.
export function greedyPolicy(engine) {
  if (engine.dead) return null;
  if (engine.player.health < HURT_BELOW) return 'retreat';
  return engine._visible().length ? 'engage' : 'explore';
}

function selfTest() {
  const d = new Doom({ seed: 3 });
  console.assert(d.player.health === 100 && d.player.ammo === 12, 'starts at full health/ammo');
  console.assert(d.enemies.length === 4, 'has 4 enemies');
  console.assert(d.candidates().join() === 'retreat,engage,explore', 'candidates are the three intents in rule order');

  const render1 = d.render();
  const lines = render1.split('\n');
  console.assert(lines.length === ROWS, 'render has ROWS lines (view and minimap share rows, ROWS === H)');
  console.assert(lines[0].length === COLS + 2 + W, 'each line is the COLS-wide view + gap + W-wide minimap row');

  // constructed situation: enemy dead ahead in the crosshair, another to the left
  const d2 = new Doom({ seed: 1 });
  d2.player = { x: 5, y: 5, h: 2, health: 60, ammo: 8 }; // facing E
  d2.enemies = [
    { x: 9, y: 5, alive: true }, // 4 ahead, on ray
    { x: 5, y: 2, alive: true }, // 3 up = "to the left" when facing E
  ];
  d2.items = [];
  d2.grid[5][6] = '.'; // ensure open ahead
  console.assert(d2.describe() === 'The player has 60 health and 8 ammo. The player has 2 enemies in sight; the nearest is 4 cells ahead in the crosshair.', 'crosshair enemy leads the sentence');
  console.assert(greedyPolicy(d2) === 'engage' && d2.resolve('engage') === 'shoot', 'engage with a crosshair target -> shoot');
  d2.enemies[0].alive = false;
  console.assert(d2.describe() === 'The player has 60 health and 8 ammo. The player has an enemy in sight, 3 cells to the left.', 'single off-axis enemy');
  console.assert(d2.resolve('engage') === 'turn left', 'engage with an enemy to the left -> turn left');
  d2.player.health = 20;
  console.assert(d2.describe().includes('The player is badly hurt.') && greedyPolicy(d2) === 'retreat', 'badly hurt -> retreat');
  console.assert(d2.resolve('retreat') === 'move back', 'retreat -> move back when open behind');
  d2.step('retreat');
  console.assert(d2.lastAction === 'move back' && d2.player.x === 4, 'step() resolves the intent and records lastAction');
  d2.step('turn right');
  console.assert(d2.lastAction === 'turn right' && d2.player.h === 3, 'raw actions (human override) pass through');

  // walled-in situation, nothing in sight
  const d3 = new Doom({ seed: 1 });
  d3.player = { x: 1, y: 1, h: 6, health: 100, ammo: 0 }; // facing W, wall ahead, open behind (E)
  d3.enemies = [];
  console.assert(d3.describe() === 'The player has 100 health and 0 ammo. The player sees no enemy. A wall is ahead.', 'no enemy + wall');
  console.assert(greedyPolicy(d3) === 'explore' && d3.resolve('explore') === 'turn right', 'explore turns when blocked');
  console.assert(!d3._legalActions().includes('shoot'), 'no shoot at 0 ammo');
  console.assert(QUESTION.includes('\nretreat: applies when the player is badly hurt  engage: '), 'QUESTION renders like query_text');

  // determinism: same seed, same greedy run -> identical final state
  function run(seed) {
    const e = new Doom({ seed });
    for (let i = 0; i < 200 && !e.dead; i++) e.step(greedyPolicy(e));
    return JSON.stringify(e.state());
  }
  console.assert(run(42) === run(42), 'same seed produces identical 200-tick greedy runs');

  // a long greedy run should not throw and should keep ticking
  const d4 = new Doom({ seed: 9 });
  for (let i = 0; i < 200 && !d4.dead; i++) d4.step(greedyPolicy(d4));
  console.assert(d4.ticks > 0, 'greedy run advances ticks');
  console.assert(d4.ticks <= 200, 'greedy run stops at death or 200 ticks');

  console.log('doom.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
