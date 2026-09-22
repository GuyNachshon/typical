// ASCII arena FPS engine. Pure state machine, no DOM, no deps - matches site/js/snake.js's
// shape: state()/describe()/render()/candidates()/step()/selfTest(). The raycast render is a
// simple ray-marcher (fixed small steps, not full DDA) - the map is 24x16, so marching is cheap
// and much easier to get right than a DDA/grid-traversal implementation.

// Demo-spec v4 grammar (50/50 on the aiming probe, 20/20 on real-engine sentences; chance .20,
// .72 with no rules): the state pre-computes the comparison each rule needs as an agent-subject
// sentence ("The player must turn left to face the enemy." - the v3 object-location form "the
// enemy is 5 cells to the left" never fired a turn, 0/12), so the model aims: shoot / turn left /
// turn right are its own labels and the engine only presses the key. `explore` must stay an
// explicit condition ("sees no enemy"): as "none of the above rules fire" it went 0/8. RULES
// order = precedence order = candidate order.
export const RULES = {
  retreat: 'applies when the player is badly hurt',
  shoot: 'applies when the player has an enemy in the crosshair',
  'turn left': 'applies when the player must turn left to face the enemy',
  'turn right': 'applies when the player must turn right to face the enemy',
  explore: 'applies when the player sees no enemy',
};
export const LABELS = Object.keys(RULES);
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
const SHOOT_RANGE = AWARE_RANGE; // "in the crosshair" == shoot() would hit
const CROSSHAIR_DEG = 22.5; // half the 8-heading step: a turn cannot line up any better than this
const DAMAGE = 10;
const FOV = Math.PI / 3;
const COLS = 72;
const ROWS = 16;
const MAX_CAST = 16;
const WALL_SHADE = ['█', '▓', '▒', '░', '·']; // near -> far

function chebyshev(x1, y1, x2, y2) {
  return Math.max(Math.abs(x1 - x2), Math.abs(y1 - y2));
}

// Angle of (ex,ey) relative to a viewer at (px,py) facing heading index h, in degrees:
// 0 = dead ahead, positive = to the right, +-180 = behind.
function angleTo(px, py, h, ex, ey) {
  const { dx, dy } = HEADINGS[h];
  const vx = ex - px;
  const vy = ey - py;
  const fwd = dx * vx + dy * vy;
  const right = -dy * vx + dx * vy;
  return (Math.atan2(right, fwd) * 180) / Math.PI;
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

  // The model's candidates: the five labels, minus `shoot` at 0 ammo (the only real illegality;
  // gating the turns on an enemy in sight was 50/50 too but costs p_null .13 vs .02).
  candidates() {
    if (this.dead) return [];
    return LABELS.filter((l) => l !== 'shoot' || this.player.ammo > 0);
  }

  // Enemies within AWARE_RANGE with a clear line, crosshair first, then nearest first. `side` is
  // the shorter turn toward the enemy (behind = right).
  _visible() {
    const p = this.player;
    return this.enemies
      .filter((e) => e.alive && chebyshev(e.x, e.y, p.x, p.y) <= AWARE_RANGE && this._clearLine(p.x, p.y, e.x, e.y))
      .map((e) => {
        const angle = angleTo(p.x, p.y, p.h, e.x, e.y);
        return { e, dist: chebyshev(e.x, e.y, p.x, p.y), crosshair: Math.abs(angle) <= CROSSHAIR_DEG, side: angle < 0 ? 'left' : 'right' };
      })
      .sort((a, b) => (a.crosshair === b.crosshair ? a.dist - b.dist : a.crosshair ? -1 : 1));
  }

  // No solid cell on the segment between two cell centres (sampled every half cell).
  _clearLine(x1, y1, x2, y2) {
    const n = Math.max(1, 2 * chebyshev(x1, y1, x2, y2));
    for (let i = 1; i < n; i++) {
      if (this._solid(Math.round(x1 + ((x2 - x1) * i) / n), Math.round(y1 + ((y2 - y1) * i) / n))) return false;
    }
    return true;
  }

  // Label -> key press. The model aims (shoot / turn left / turn right are its own labels); the
  // engine only fills in the walking for retreat/explore. Raw actions pass through so the
  // human override keeps working.
  resolve(move) {
    const legal = this._legalActions();
    const pick = (...prefs) => prefs.find((a) => legal.includes(a)) ?? legal[0] ?? null;
    if (move === 'retreat') return pick('move back', 'turn right');
    if (move === 'explore') return pick('move forward', 'turn right');
    return move;
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

  // The nearest enemy inside the crosshair cone (+-CROSSHAIR_DEG, <= SHOOT_RANGE, clear line) -
  // exactly the enemy describe() calls "in the crosshair".
  _shootTarget() {
    const t = this._visible().find((v) => v.crosshair && v.dist <= SHOOT_RANGE);
    return t ? t.e : null;
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

  // Situations-only grammar (demo-spec v4): health/ammo, then only the situations that apply,
  // worded exactly as the rule conditions, in this fixed order. The leading enemy (crosshair
  // first, then nearest) is the one the sentence is about; the turn side is pre-computed.
  describe() {
    const p = this.player;
    const sentences = [`The player has ${p.health} health and ${p.ammo} ammo.`];
    if (p.health < HURT_BELOW) sentences.push('The player is badly hurt.');
    const seen = this._visible();
    if (seen.length > 1) sentences.push(`The player has ${seen.length} enemies in sight.`);
    const who = seen.length > 1 ? 'the nearest enemy' : 'the enemy';
    if (!seen.length) sentences.push('The player sees no enemy.');
    else if (seen[0].crosshair) sentences.push(`The player has an enemy in the crosshair, ${seen[0].dist} cell${seen[0].dist === 1 ? '' : 's'} ahead.`);
    else sentences.push(`The player must turn ${seen[0].side} to face ${who}.`);
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

// Scripted policy = RULES applied literally (first matching rule wins) over the offered labels;
// step() resolves it to a key press. This is the gold the model is measured against. With an
// enemy in the crosshair and no ammo nothing fires; the residual goes to the first label.
export function greedyPolicy(engine) {
  if (engine.dead) return null;
  if (engine.player.health < HURT_BELOW) return 'retreat';
  const lead = engine._visible()[0];
  if (!lead) return 'explore';
  if (lead.crosshair) return engine.player.ammo > 0 ? 'shoot' : 'retreat';
  return `turn ${lead.side}`;
}

function selfTest() {
  const d = new Doom({ seed: 3 });
  console.assert(d.player.health === 100 && d.player.ammo === 12, 'starts at full health/ammo');
  console.assert(d.enemies.length === 4, 'has 4 enemies');
  console.assert(d.candidates().join() === 'retreat,shoot,turn left,turn right,explore', 'candidates are the five labels in rule order');

  const render1 = d.render();
  const lines = render1.split('\n');
  console.assert(lines.length === ROWS, 'render has ROWS lines (view and minimap share rows, ROWS === H)');
  console.assert(lines[0].length === COLS + 2 + W, 'each line is the COLS-wide view + gap + W-wide minimap row');

  // constructed situation: enemy dead ahead in the crosshair, another to the left
  const d2 = new Doom({ seed: 1 });
  d2.player = { x: 5, y: 5, h: 2, health: 60, ammo: 8 }; // facing E
  d2.enemies = [
    { x: 9, y: 5, alive: true }, // 4 ahead, in the crosshair
    { x: 5, y: 2, alive: true }, // 3 up = to the left when facing E
  ];
  d2.items = [];
  d2.grid[5][6] = '.'; // ensure open ahead
  console.assert(d2.describe() === 'The player has 60 health and 8 ammo. The player has 2 enemies in sight. The player has an enemy in the crosshair, 4 cells ahead.', 'crosshair enemy leads the sentence');
  console.assert(greedyPolicy(d2) === 'shoot' && d2.resolve('shoot') === 'shoot' && d2._shootTarget() === d2.enemies[0], 'crosshair target -> shoot hits it');
  d2.enemies[0].alive = false;
  console.assert(d2.describe() === 'The player has 60 health and 8 ammo. The player must turn left to face the enemy.', 'single off-axis enemy -> pre-computed turn side');
  console.assert(greedyPolicy(d2) === 'turn left' && d2.resolve('turn left') === 'turn left', 'the model turns; the engine presses the key');
  d2.enemies[1].y = 7; // 2 down, 0 across = to the right when facing E
  console.assert(greedyPolicy(d2) === 'turn right', 'right side');
  d2.enemies[1] = { x: 8, y: 4, alive: true }; // 3 ahead, 1 up: angle -18 deg -> inside the 22.5 deg cone
  console.assert(d2.describe().includes('in the crosshair, 3 cells ahead') && d2._shootTarget() === d2.enemies[1], 'cone crosshair: 8-heading resolution');
  d2.enemies[1] = { x: 1, y: 5, alive: true }; // behind
  console.assert(greedyPolicy(d2) === 'turn right', 'behind -> turn right (the shorter side is a tie; engine picks right)');
  d2.player.ammo = 0;
  d2.enemies[1] = { x: 9, y: 5, alive: true };
  console.assert(!d2.candidates().includes('shoot') && greedyPolicy(d2) === 'retreat', 'no ammo: shoot not offered, crosshair residual -> retreat');
  d2.player.ammo = 8;
  d2.player.health = 20;
  console.assert(d2.describe().includes('The player is badly hurt.') && greedyPolicy(d2) === 'retreat', 'badly hurt -> retreat');
  console.assert(d2.resolve('retreat') === 'move back', 'retreat -> move back when open behind');
  d2.step('retreat');
  console.assert(d2.lastAction === 'move back' && d2.player.x === 4, 'step() resolves the label and records lastAction');
  d2.step('turn right');
  console.assert(d2.lastAction === 'turn right' && d2.player.h === 3, 'raw actions (human override) pass through');

  // walled-in situation, nothing in sight
  const d3 = new Doom({ seed: 1 });
  d3.player = { x: 1, y: 1, h: 6, health: 100, ammo: 0 }; // facing W, wall ahead, open behind (E)
  d3.enemies = [];
  console.assert(d3.describe() === 'The player has 100 health and 0 ammo. The player sees no enemy. A wall is ahead.', 'no enemy + wall');
  console.assert(greedyPolicy(d3) === 'explore' && d3.resolve('explore') === 'turn right', 'explore turns when blocked');
  console.assert(!d3._legalActions().includes('shoot'), 'no shoot at 0 ammo');
  // an enemy behind a wall is not in sight
  d3.player = { x: 5, y: 5, h: 2, health: 100, ammo: 5 };
  d3.enemies = [{ x: 15, y: 5, alive: true }]; // the '#' column at x=10 is between
  console.assert(d3.describe().includes('sees no enemy') && d3._shootTarget() === null, 'wall blocks sight and shots');
  console.assert(QUESTION.includes('\nretreat: applies when the player is badly hurt  shoot: '), 'QUESTION renders like query_text');

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
