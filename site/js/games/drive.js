// Top-down ASCII driving engine (JevPilot-style: text state in, text state out). Pure state
// machine, no DOM, no deps - matches site/js/snake.js's shape: state()/describe()/render()/
// candidates()/step()/selfTest().

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ACTIONS = ['hold speed', 'accelerate', 'brake', 'change lane left', 'change lane right', 'stop'];
const LANES = 3;
const ROAD_LENGTH = 2000; // m
const DEST_POS = 1900; // m - exit requires lane 3
const LIGHT_SPACING = 300; // m
const TICK_S = 0.5;
const MAX_SPEED = 80; // km/h
const ACCEL = 8; // km/h per tick
const BRAKE = 15; // km/h per tick
const STOP_DECEL = 30; // km/h per tick
const LANE_CHANGE_GAP = 5; // m - min clearance in target lane
const AWARE_RANGE = 100; // m - describe()/render visibility
const COLLISION_GAP = 3; // m - ponytail: treat "gap <= 0" as "within a car length", since a
// 0.5s tick can step two cars past each other without ever landing on an exact 0 gap.
const LIGHT_GREEN_TICKS = 10;
const LIGHT_RED_TICKS = 10;

function metersPerKmh(kmh) {
  return (kmh / 3.6) * TICK_S;
}

export class Drive {
  constructor({ seed = 1 } = {}) {
    this.rng = mulberry32(seed);
    this.reset();
  }

  reset() {
    this.ego = { lane: 2, speed: 40, position: 0 };
    this.lights = [];
    for (let pos = LIGHT_SPACING; pos < DEST_POS; pos += LIGHT_SPACING) {
      const state = this.rng() < 0.5 ? 'red' : 'green';
      const timer = 1 + Math.floor(this.rng() * (state === 'red' ? LIGHT_RED_TICKS : LIGHT_GREEN_TICKS));
      this.lights.push({ pos, state, timer });
    }
    this.traffic = [];
    for (let i = 0; i < 7; i++) {
      this.traffic.push({
        lane: 1 + Math.floor(this.rng() * LANES),
        position: 80 + this.rng() * (ROAD_LENGTH - 160),
        speed: 20 + Math.floor(this.rng() * 45),
      });
    }
    this.pedestrians = [];
    this.distance = 0;
    this.violations = 0;
    this.collisions = 0;
    this.arrived = false;
    this.missedExit = false;
    this.done = false;
    this.ticks = 0;
    return this.state();
  }

  state() {
    return {
      ego: { ...this.ego },
      lights: this.lights.map((l) => ({ ...l })),
      traffic: this.traffic.map((c) => ({ ...c })),
      pedestrians: this.pedestrians.map((p) => ({ ...p })),
      distance: this.distance,
      violations: this.violations,
      collisions: this.collisions,
      arrived: this.arrived,
      missedExit: this.missedExit,
      done: this.done,
      ticks: this.ticks,
    };
  }

  _laneClear(lane) {
    return !this.traffic.some((c) => c.lane === lane && Math.abs(c.position - this.ego.position) <= LANE_CHANGE_GAP);
  }

  candidates() {
    if (this.done) return [];
    const legal = new Set(['hold speed']);
    if (this.ego.speed < MAX_SPEED) legal.add('accelerate');
    if (this.ego.speed > 0) legal.add('brake');
    if (this.ego.lane > 1 && this._laneClear(this.ego.lane - 1)) legal.add('change lane left');
    if (this.ego.lane < LANES && this._laneClear(this.ego.lane + 1)) legal.add('change lane right');
    if (this.ego.speed > 0) legal.add('stop');
    return ACTIONS.filter((a) => legal.has(a));
  }

  safeMoves() {
    return this.candidates();
  }

  _applyAction(action) {
    const e = this.ego;
    if (action === 'accelerate') e.speed = Math.min(MAX_SPEED, e.speed + ACCEL);
    else if (action === 'brake') e.speed = Math.max(0, e.speed - BRAKE);
    else if (action === 'stop') e.speed = Math.max(0, e.speed - STOP_DECEL);
    else if (action === 'change lane left') e.lane -= 1;
    else if (action === 'change lane right') e.lane += 1;
    // 'hold speed' and unrecognised/illegal actions: no-op.
  }

  step(action) {
    if (this.done) return this.state();
    if (this.candidates().includes(action)) this._applyAction(action);

    const lightStatesAtCrossing = this.lights.map((l) => l.state);
    const prevPos = this.ego.position;
    this.ego.position += metersPerKmh(this.ego.speed);
    for (const car of this.traffic) car.position += metersPerKmh(car.speed);

    for (const l of this.lights) {
      l.timer -= 1;
      if (l.timer <= 0) {
        l.state = l.state === 'red' ? 'green' : 'red';
        l.timer = l.state === 'red' ? LIGHT_RED_TICKS : LIGHT_GREEN_TICKS;
      }
    }

    // Pedestrians occasionally cross near a light close ahead; a handful of ticks later they clear.
    this.pedestrians = this.pedestrians.filter((p) => --p.ticksLeft > 0);
    for (const l of this.lights) {
      const ahead = l.pos - this.ego.position;
      if (ahead < 0 || ahead > 40) continue;
      if (this.pedestrians.some((p) => Math.abs(p.pos - l.pos) < 10)) continue;
      if (this.rng() < 0.04) {
        this.pedestrians.push({ pos: l.pos, lane: 1 + Math.floor(this.rng() * LANES), ticksLeft: 4 });
      }
    }

    this.lights.forEach((l, i) => {
      if (prevPos < l.pos && l.pos <= this.ego.position && lightStatesAtCrossing[i] === 'red') {
        this.violations += 1;
      }
    });

    if (this.traffic.some((c) => c.lane === this.ego.lane && Math.abs(c.position - this.ego.position) <= COLLISION_GAP)) {
      this.collisions += 1;
      this.done = true;
    }

    if (prevPos < DEST_POS && DEST_POS <= this.ego.position) {
      if (this.ego.lane === LANES) this.arrived = true;
      else this.missedExit = true;
      this.done = true;
    }

    this.distance = this.ego.position;
    this.ticks += 1;
    return this.state();
  }

  // Fact-sentence grammar: ego line, own-lane lead vehicle (or explicit "clear"), light ahead
  // (only within AWARE_RANGE), each other lane's nearest vehicle (or "clear"), destination,
  // then a fixed-order "Legal actions:" sentence.
  describe() {
    const e = this.ego;
    const sentences = [`The car is in lane ${e.lane} of ${LANES} at ${e.speed} km/h.`];

    const lead = this.traffic
      .filter((c) => c.lane === e.lane && c.position > e.position)
      .sort((a, b) => a.position - b.position)[0];
    const leadDist = lead ? lead.position - e.position : Infinity;
    if (lead && leadDist <= AWARE_RANGE) {
      sentences.push(`The lead vehicle is ${Math.round(leadDist)} m ahead at ${lead.speed} km/h.`);
    } else {
      sentences.push('No vehicle ahead within 100 m.');
    }

    const light = this.lights
      .filter((l) => l.pos >= e.position && l.pos - e.position <= AWARE_RANGE)
      .sort((a, b) => a.pos - b.pos)[0];
    if (light) sentences.push(`The traffic light ${Math.round(light.pos - e.position)} m ahead is ${light.state}.`);

    for (let lane = 1; lane <= LANES; lane++) {
      if (lane === e.lane) continue;
      const nearest = this.traffic
        .filter((c) => c.lane === lane && Math.abs(c.position - e.position) <= AWARE_RANGE)
        .sort((a, b) => Math.abs(a.position - e.position) - Math.abs(b.position - e.position))[0];
      if (!nearest) {
        sentences.push(`Lane ${lane} is clear.`);
      } else {
        const dist = Math.round(Math.abs(nearest.position - e.position));
        const bearing = nearest.position > e.position ? 'ahead' : 'behind';
        sentences.push(`Lane ${lane} has a vehicle ${dist} m ${bearing}.`);
      }
    }

    sentences.push(`The destination is an exit on the right in ${Math.max(0, Math.round(DEST_POS - e.position))} m.`);
    sentences.push(`Legal actions: ${this.candidates().join(', ')}.`);
    return sentences.join(' ');
  }

  render() {
    const ROWS = 28;
    const EGO_ROW = ROWS - 5;
    const METERS_PER_ROW = 5;
    const LANE_W = 7;
    const RAMP_WINDOW = 150; // m - exit ramp widens over this final stretch
    const e = this.ego;

    const rowFor = (pos) => EGO_ROW - Math.round((pos - e.position) / METERS_PER_ROW);
    const cell = (s) => {
      s = String(s);
      const pad = LANE_W - s.length;
      const left = Math.floor(pad / 2);
      return ' '.repeat(Math.max(0, left)) + s + ' '.repeat(Math.max(0, pad - left));
    };
    const bar = (label) => {
      const text = ` ${label} `;
      const width = LANE_W * LANES + (LANES - 1); // full width between the outer edges
      const pad = Math.max(0, width - text.length);
      const left = Math.floor(pad / 2);
      return '━'.repeat(left) + text + '━'.repeat(pad - left);
    };

    const lanes = Array.from({ length: ROWS }, (_, row) => {
      const cells = [];
      for (let lane = 1; lane <= LANES; lane++) cells.push(row % 3 === 1 ? cell('¦') : cell(''));
      return cells;
    });
    const lightRow = new Array(ROWS).fill(null);
    const rampWidth = new Array(ROWS).fill(0);

    for (const l of this.lights) {
      const row = rowFor(l.pos);
      if (row < 0 || row >= ROWS) continue;
      lightRow[row] = bar(l.state === 'red' ? 'RED' : 'GREEN');
    }
    for (const c of this.traffic) {
      const row = rowFor(c.position);
      if (row < 0 || row >= ROWS || lightRow[row]) continue;
      lanes[row][c.lane - 1] = cell('▛█▜');
    }
    for (const p of this.pedestrians) {
      const row = rowFor(p.pos);
      if (row < 0 || row >= ROWS || lightRow[row]) continue;
      lanes[row][p.lane - 1] = cell('☺');
    }
    if (e.position < DEST_POS) {
      for (let row = 0; row < ROWS; row++) {
        const pos = e.position - (row - EGO_ROW) * METERS_PER_ROW;
        const untilExit = DEST_POS - pos;
        if (untilExit >= 0 && untilExit <= RAMP_WINDOW) {
          rampWidth[row] = 1 + Math.round((4 * (RAMP_WINDOW - untilExit)) / RAMP_WINDOW);
        }
      }
    }
    if (EGO_ROW >= 0 && EGO_ROW < ROWS && !lightRow[EGO_ROW]) lanes[EGO_ROW][e.lane - 1] = cell('▟█▙');

    const rows = lanes.map((cells, row) => {
      if (lightRow[row]) return '║' + lightRow[row] + '║';
      return '║' + cells.join('┆') + '║' + '╱'.repeat(rampWidth[row]);
    });
    const hud = `${e.speed} km/h · lane ${e.lane} · exit in ${Math.max(0, Math.round(DEST_POS - e.position))} m`;
    return [hud, ...rows].join('\n');
  }
}

// Scripted policy: brake for a red light or a slower lead car close ahead; steer toward the
// exit lane (3) as the destination nears; otherwise cruise, accelerating up to 60 km/h.
export function greedyPolicy(engine) {
  const legal = engine.candidates();
  if (legal.length === 0) return null;
  const e = engine.ego;

  const lightAhead = engine.lights.find((l) => l.pos >= e.position && l.pos - e.position <= 60 && l.state === 'red');
  const lead = engine.traffic
    .filter((c) => c.lane === e.lane && c.position > e.position && c.position - e.position <= 25)
    .sort((a, b) => a.position - b.position)[0];
  if ((lightAhead || (lead && lead.speed < e.speed)) && legal.includes('brake')) return 'brake';

  const distToExit = DEST_POS - e.position;
  if (distToExit > 0 && distToExit < 200 && e.lane < LANES && legal.includes('change lane right')) {
    return 'change lane right';
  }

  if (e.speed < 60 && legal.includes('accelerate')) return 'accelerate';
  return legal.includes('hold speed') ? 'hold speed' : legal[0];
}

function selfTest() {
  const d = new Drive({ seed: 4 });
  console.assert(d.ego.lane === 2 && d.ego.speed === 40, 'ego starts in lane 2 at 40 km/h');
  console.assert(d.lights.length > 0, 'has traffic lights');
  console.assert(d.traffic.length === 7, 'has 7 traffic cars');

  const r = d.render();
  const rows = r.split('\n');
  console.assert(rows.length === 29, 'render is 1 HUD line + 28 road rows');
  console.assert(/km\/h · lane \d · exit in \d+ m/.test(rows[0]), 'HUD line matches spec format');
  console.assert(rows[1].startsWith('║') && rows[1].endsWith('║') && rows[1].includes('┆'), 'road rows use ║ edges and ┆ lane separators');
  console.assert(rows[1].length === 7 * 3 + 2 + 2, 'road row width is 3 lanes of 7 + 2 separators + 2 edges');

  // constructed situation matching the spec's example shape
  const d2 = new Drive({ seed: 1 });
  d2.ego = { lane: 2, speed: 45, position: 1000 };
  d2.traffic = [
    { lane: 2, position: 1018, speed: 30 }, // 18 m ahead in own lane
    { lane: 3, position: 995, speed: 40 }, // 5 m behind in lane 3
  ];
  d2.lights = [{ pos: 1060, state: 'red', timer: 5 }];
  const desc = d2.describe();
  console.assert(desc.startsWith('The car is in lane 2 of 3 at 45 km/h.'), 'describe starts with lane/speed sentence');
  console.assert(desc.includes('The lead vehicle is 18 m ahead at 30 km/h.'), 'reports own-lane lead vehicle');
  console.assert(desc.includes('The traffic light 60 m ahead is red.'), 'reports light ahead within range');
  console.assert(desc.includes('Lane 1 is clear.'), 'reports clear lane explicitly');
  console.assert(desc.includes('Lane 3 has a vehicle 5 m behind.'), 'reports nearest vehicle in other lane with bearing');
  console.assert(desc.includes('The destination is an exit on the right in 900 m.'), 'reports destination distance');
  console.assert(!d2.candidates().includes('change lane right'), 'lane change blocked by a car within 5 m');

  // legality: no lane change off the road
  const d3 = new Drive({ seed: 1 });
  d3.ego.lane = 1;
  console.assert(!d3.candidates().includes('change lane left'), 'cannot change lane left off the road');
  d3.ego.lane = 3;
  console.assert(!d3.candidates().includes('change lane right'), 'cannot change lane right off the road');

  // determinism
  function run(seed) {
    const e = new Drive({ seed });
    for (let i = 0; i < 200 && !e.done; i++) e.step(greedyPolicy(e));
    return JSON.stringify(e.state());
  }
  console.assert(run(11) === run(11), 'same seed produces identical 200-tick greedy runs');

  // long greedy run without crashing
  const d4 = new Drive({ seed: 2 });
  for (let i = 0; i < 200 && !d4.done; i++) d4.step(greedyPolicy(d4));
  console.assert(d4.ticks > 0, 'greedy run advances ticks');
  console.assert(d4.ticks <= 200, 'greedy run stops at done or 200 ticks');

  console.log('drive.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
