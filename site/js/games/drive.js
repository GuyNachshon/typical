// Top-down ASCII driving engine (JevPilot-style: text state in, text state out). Pure state
// machine, no DOM, no deps - matches site/js/snake.js's shape: state()/describe()/render()/
// candidates()/step()/selfTest().

// Demo-spec v3 grammar (70/70 on the probe set, chance .16): the state lists only the situations
// that apply, phrased exactly as the rule conditions; one condition per rule; the candidate set
// is the legality guardrail (a lane change is only offered when that lane exists and is clear).
// RULES order = precedence order = candidate order.
export const RULES = {
  stop: "applies when a pedestrian is crossing the car's lane",
  brake: 'applies when the traffic light ahead is red',
  'swerve left': "applies when a cone blocks the car's lane ahead", // visitor-placed hazard (site interactivity)
  'change lane right': 'applies when the car must take the exit on the right',
  accelerate: 'applies when the road ahead is clear for 50 m', // speeds up to the limit; no-op at it
  'change lane left': 'applies when the car is closing on a slower vehicle ahead',
  follow: 'applies when the car is closing on a slower vehicle ahead',
  'hold speed': 'applies when none of the above rules fire',
};
// Rendered exactly like inference/typical/core.py::query_text (instructions + "\n" + "label: desc"
// pairs joined by two spaces); sent verbatim as the query by render-drive.js / record_games.mjs.
export const QUESTION =
  'Which manoeuvre applies for driving the car? Apply the rules in the stated precedence order; the first matching rule wins.\n' +
  Object.entries(RULES).map(([k, v]) => `${k}: ${v}`).join('  ');

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ACTIONS = Object.keys(RULES);
const LANES = 3;
export const ROAD_LENGTH = 1600; // m
export const DEST_POS = 1500; // m - exit requires lane 3 (render-drive.js draws the ramp here)
const LIGHT_SPACING = 300; // m
const TICK_S = 0.5;
const SPEED_LIMIT = 60; // km/h - accelerate caps here
const ACCEL = 8; // km/h per tick
const BRAKE = 15; // km/h per tick
const STOP_DECEL = 30; // km/h per tick
const LANE_CLEAR_GAP = 15; // m - a lane is "clear" with no vehicle within this, ahead or behind
const PED_RANGE = 20; // m - "a pedestrian is crossing the car's lane"
const LIGHT_RANGE = 60; // m - "the traffic light ahead is red"
const CLOSING_RANGE = 20; // m - "closing on a slower vehicle ahead": slower lead within this
const LEAD_RANGE = 50; // m - lead vehicle reported (else "the road ahead is clear for 50 m")
const EXIT_RANGE = 300; // m - "the car must take the exit on the right"
const TRAFFIC_GAP = 10; // m - traffic cars wait behind anything this close, and at red lights
const COLLISION_GAP = 3; // m - ponytail: treat "gap <= 0" as "within a car length", since a
// 0.5s tick can step two cars past each other without ever landing on an exact 0 gap.
const LIGHT_GREEN_TICKS = 20;
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
    for (let i = 0; i < 16; i++) {
      const lane = 1 + Math.floor(this.rng() * LANES);
      // Everyone is slower than the 60 km/h limit, so the ego catches up and has to overtake
      // (left only, per RULES); the left lane is the fast one so it is not stuck there for good.
      const cruise = [40, 25, 15][lane - 1] + Math.floor(this.rng() * 11);
      this.traffic.push({
        lane,
        position: 60 + this.rng() * 600, // bunched in the first 660 m so the ego catches them
        speed: cruise,
        cruise,
      });
    }
    this.pedestrians = [];
    this.cones = [];
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
      cones: this.cones.map((c) => ({ ...c })),
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
    return !this.traffic.some((c) => c.lane === lane && Math.abs(c.position - this.ego.position) <= LANE_CLEAR_GAP);
  }

  _lead() {
    const e = this.ego;
    return this.traffic
      .filter((c) => c.lane === e.lane && c.position > e.position && c.position - e.position <= LEAD_RANGE)
      .sort((a, b) => a.position - b.position)[0] ?? null;
  }

  // The situations describe() reports and candidates() gates on - one boolean per rule condition.
  situations() {
    const e = this.ego;
    const lead = this._lead();
    const light = this.lights.find((l) => l.pos >= e.position && l.pos - e.position <= LIGHT_RANGE);
    const ped = this.pedestrians.find((p) => p.lane === e.lane && p.pos >= e.position && p.pos - e.position <= PED_RANGE);
    const cone = this.cones.find((c) => c.lane === e.lane && c.pos >= e.position && c.pos - e.position <= PED_RANGE);
    const exitIn = DEST_POS - e.position;
    return {
      lead,
      light,
      ped,
      cone,
      exitIn,
      red: Boolean(light && light.state === 'red'),
      coneAhead: Boolean(cone),
      closing: Boolean(lead && lead.speed < e.speed && lead.position - e.position <= CLOSING_RANGE),
      exitNear: exitIn > 0 && exitIn <= EXIT_RANGE,
      mustExit: exitIn > 0 && exitIn <= EXIT_RANGE && e.lane < LANES,
      roadClear: !lead,
      belowLimit: e.speed < SPEED_LIMIT,
      leftClear: e.lane > 1 && this._laneClear(e.lane - 1),
      rightClear: e.lane < LANES && this._laneClear(e.lane + 1),
    };
  }

  // Legality guardrail (the page states it): lane changes only into an existing clear lane, and
  // only toward the exit once it is within EXIT_RANGE. Everything else is always offered
  // (no-ops at 0 km/h, at the limit, or with no lead vehicle) - gating accelerate at the limit
  // made the model (correctly) answer null on every cruise tick: mean p_null .49 vs .23.
  candidates() {
    if (this.done) return [];
    const sit = this.situations();
    return ACTIONS.filter((a) => {
      if (a === 'change lane right') return sit.rightClear;
      if (a === 'change lane left') return sit.leftClear && !sit.exitNear;
      if (a === 'swerve left') return sit.leftClear;
      return true;
    });
  }

  safeMoves() {
    return this.candidates();
  }

  // Visitor-placed hazard (site interactivity): validated on-road, ahead-of-the-car placement
  // that lands in the same arrays procedurally spawned objects use, so it flows through the
  // existing situations()/describe() pipeline - the model always reads one of the pre-computed
  // sentences above, never a raw coordinate. Rejects (returns false, no mutation) anything the
  // engine can't represent: off-road lane, or behind/at the car.
  placeHazard(kind, lane, pos) {
    if (this.done || lane < 1 || lane > LANES || pos <= this.ego.position || pos >= ROAD_LENGTH) return false;
    if (kind === 'pedestrian') this.pedestrians.push({ pos, lane, ticksLeft: 20 }); // longer-lived than a
    // spawned pedestrian's 4 ticks - a visitor-placed one should stick around long enough to watch.
    else if (kind === 'cone') this.cones.push({ pos, lane });
    else if (kind === 'car') this.traffic.push({ lane, position: pos, speed: 0, cruise: 0 });
    else return false;
    return true;
  }

  _applyAction(action) {
    const e = this.ego;
    if (action === 'accelerate') e.speed = Math.min(SPEED_LIMIT, e.speed + ACCEL);
    else if (action === 'brake') e.speed = Math.max(0, e.speed - BRAKE);
    else if (action === 'stop') e.speed = Math.max(0, e.speed - STOP_DECEL);
    else if (action === 'follow') {
      const lead = this._lead();
      if (lead) e.speed = Math.max(0, Math.min(e.speed, lead.speed));
    } else if (action === 'change lane left' || action === 'swerve left') e.lane -= 1;
    else if (action === 'change lane right') e.lane += 1;
    // 'hold speed' and unrecognised/illegal actions: no-op.
  }

  step(action) {
    if (this.done) return this.state();
    if (this.candidates().includes(action)) this._applyAction(action);

    const lightStatesAtCrossing = this.lights.map((l) => l.state);
    const prevPos = this.ego.position;
    this.ego.position += metersPerKmh(this.ego.speed);
    // Traffic is dumb but not suicidal: a car matches whatever is within TRAFFIC_GAP ahead in its
    // lane (the ego included) and stops at a red light, so a stopped ego is not rammed from behind.
    // car.speed is the speed it actually moves at this tick (what describe() reports); car.cruise
    // is its own preferred speed.
    for (const car of this.traffic) {
      const ahead = [...this.traffic, this.ego]
        .filter((o) => o !== car && o.lane === car.lane && o.position > car.position && o.position - car.position <= TRAFFIC_GAP)
        .sort((a, b) => a.position - b.position)[0];
      const redAhead = this.lights.some((l) => l.state === 'red' && l.pos >= car.position && l.pos - car.position <= TRAFFIC_GAP);
      car.speed = redAhead ? 0 : ahead ? Math.min(car.cruise, ahead.speed) : car.cruise;
      car.position += metersPerKmh(car.speed);
    }

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

  // Situations-only grammar (demo-spec v3): the ego line, then one sentence per situation that
  // applies, worded exactly as its rule condition, in this fixed order. Sentence order matters to
  // the model (shuffled orders scored .89-.93 vs 1.00 on the probe set) - keep it.
  describe() {
    const e = this.ego;
    const sit = this.situations();
    const sentences = [
      `The car is in lane ${e.lane} of ${LANES} at ${e.speed} km/h, ${sit.belowLimit ? 'below' : 'at'} the speed limit of ${SPEED_LIMIT} km/h.`,
    ];
    if (sit.ped) sentences.push(`A pedestrian is crossing the car's lane ${Math.round(sit.ped.pos - e.position)} m ahead.`);
    if (sit.red) sentences.push(`The traffic light ${Math.round(sit.light.pos - e.position)} m ahead is red.`);
    if (sit.coneAhead) sentences.push(`A cone blocks the car's lane ${Math.round(sit.cone.pos - e.position)} m ahead.`);
    if (sit.closing) sentences.push(`The car is closing on a slower vehicle ${Math.round(sit.lead.position - e.position)} m ahead.`);
    else if (sit.lead) sentences.push(`The lead vehicle is ${Math.round(sit.lead.position - e.position)} m ahead at ${sit.lead.speed} km/h.`);
    if (sit.mustExit) sentences.push(`The car must take the exit on the right in ${Math.round(sit.exitIn)} m.`);
    if (sit.roadClear) sentences.push(`The road ahead is clear for ${LEAD_RANGE} m.`);
    if (sit.leftClear) sentences.push('The lane to the left is clear.');
    if (sit.rightClear) sentences.push('The lane to the right is clear.');
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
    for (const c of this.cones) {
      const row = rowFor(c.pos);
      if (row < 0 || row >= ROWS || lightRow[row]) continue;
      lanes[row][c.lane - 1] = cell('▲');
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

// Scripted policy = RULES applied literally, first matching offered rule wins. This is the gold
// the model is measured against (record_games.mjs reports agreement with it).
export function greedyPolicy(engine) {
  const legal = engine.candidates();
  if (legal.length === 0) return null;
  const sit = engine.situations();
  const fires = {
    stop: Boolean(sit.ped),
    brake: sit.red,
    'swerve left': sit.coneAhead,
    'change lane right': sit.mustExit,
    accelerate: sit.roadClear,
    'change lane left': sit.closing,
    follow: sit.closing,
    'hold speed': true,
  };
  return legal.find((a) => fires[a]);
}

function selfTest() {
  const d = new Drive({ seed: 4 });
  console.assert(d.ego.lane === 2 && d.ego.speed === 40, 'ego starts in lane 2 at 40 km/h');
  console.assert(d.candidates().length >= 5 && d.candidates()[0] === 'stop', 'candidates in rule order');
  console.assert(d.lights.length > 0, 'has traffic lights');
  console.assert(d.traffic.length === 16, "has 16 traffic cars");

  const r = d.render();
  const rows = r.split('\n');
  console.assert(rows.length === 29, 'render is 1 HUD line + 28 road rows');
  console.assert(/km\/h · lane \d · exit in \d+ m/.test(rows[0]), 'HUD line matches spec format');
  console.assert(rows[1].startsWith('║') && rows[1].endsWith('║') && rows[1].includes('┆'), 'road rows use ║ edges and ┆ lane separators');
  console.assert(rows[1].length === 7 * 3 + 2 + 2, 'road row width is 3 lanes of 7 + 2 separators + 2 edges');

  // constructed situation: closing on a slower lead, left clear, right blocked, red light ahead
  const d2 = new Drive({ seed: 1 });
  d2.ego = { lane: 2, speed: 45, position: 1000 };
  d2.traffic = [
    { lane: 2, position: 1018, speed: 30 }, // 18 m ahead in own lane, slower
    { lane: 3, position: 995, speed: 40 }, // 5 m behind in lane 3 -> not clear
  ];
  d2.lights = [{ pos: 1060, state: 'red', timer: 5 }];
  d2.pedestrians = [];
  console.assert(
    d2.describe() ===
      'The car is in lane 2 of 3 at 45 km/h, below the speed limit of 60 km/h. The traffic light 60 m ahead is red. The car is closing on a slower vehicle 18 m ahead. The lane to the left is clear.',
    'describe() lists only the situations that apply, in the fixed order'
  );
  console.assert(!d2.candidates().includes('change lane right'), 'lane change blocked by a car within 15 m');
  console.assert(d2.candidates().includes('change lane left'), 'clear lane offered');
  console.assert(greedyPolicy(d2) === 'brake', 'red light outranks the lane change');
  d2.lights = [];
  console.assert(greedyPolicy(d2) === 'change lane left', 'closing on a slower vehicle with a clear left lane -> change lane left');
  d2.traffic.push({ lane: 1, position: 1005, speed: 40 });
  console.assert(greedyPolicy(d2) === 'follow', 'no clear lane -> follow');
  d2.step('follow');
  console.assert(d2.ego.speed === 30, 'follow matches the lead speed');

  // exit + clear road + pedestrian
  const d5 = new Drive({ seed: 1 });
  d5.ego = { lane: 2, speed: 40, position: 1300 };
  d5.traffic = [];
  d5.lights = [];
  d5.pedestrians = [];
  console.assert(d5.describe() === 'The car is in lane 2 of 3 at 40 km/h, below the speed limit of 60 km/h. The car must take the exit on the right in 200 m. The road ahead is clear for 50 m. The lane to the left is clear. The lane to the right is clear.', 'exit sentence precedes the clear-road sentence');
  console.assert(!d5.candidates().includes('change lane left'), 'no lane change away from the exit within 300 m');
  console.assert(greedyPolicy(d5) === 'change lane right', 'exit outranks accelerate');
  d5.pedestrians = [{ pos: 1312, lane: 2, ticksLeft: 3 }];
  console.assert(d5.describe().includes("A pedestrian is crossing the car's lane 12 m ahead."), 'pedestrian sentence');
  console.assert(greedyPolicy(d5) === 'stop', 'pedestrian outranks everything');
  d5.ego.speed = 60;
  d5.pedestrians = [];
  d5.step('accelerate');
  console.assert(d5.ego.speed === 60, 'accelerate is a no-op at the limit');
  console.assert(QUESTION.startsWith('Which manoeuvre applies') && QUESTION.includes('\nstop: applies when') && QUESTION.includes('  hold speed: applies when none of the above rules fire'), 'QUESTION renders like query_text');

  // legality: no lane change off the road
  const d3 = new Drive({ seed: 1 });
  d3.ego.lane = 1;
  console.assert(!d3.candidates().includes('change lane left'), 'cannot change lane left off the road');
  d3.ego.lane = 3;
  console.assert(!d3.candidates().includes('change lane right'), 'cannot change lane right off the road');

  // visitor-placed hazards flow through the same pipeline as spawned ones
  const d6 = new Drive({ seed: 1 });
  d6.ego = { lane: 2, speed: 40, position: 500 };
  d6.traffic = [];
  d6.lights = [];
  d6.pedestrians = [];
  d6.cones = [];
  console.assert(!d6.placeHazard('cone', 2, 480), 'hazard behind the car is rejected');
  console.assert(!d6.placeHazard('cone', 4, 520), 'hazard off-road is rejected');
  console.assert(d6.placeHazard('cone', 2, 515), 'on-road hazard ahead of the car is accepted');
  console.assert(d6.describe().includes("A cone blocks the car's lane 15 m ahead."), 'placed cone renders the same sentence style as a spawned hazard');
  console.assert(d6.candidates().includes('swerve left'), 'swerve left is offered with the left lane clear');
  console.assert(greedyPolicy(d6) === 'swerve left', 'cone in the lane outranks the default');
  console.assert(d6.placeHazard('pedestrian', 1, 510), 'pedestrian hazard accepted ahead in a different lane');
  console.assert(d6.placeHazard('car', 3, 530), 'stopped-car hazard accepted (reuses the traffic array)');
  console.assert(d6.traffic.length === 1 && d6.traffic[0].speed === 0, 'stopped-car hazard has zero speed');

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
