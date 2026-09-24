// Top-down ASCII driving engine (JevPilot-style: text state in, text state out). Pure state
// machine, no DOM, no deps - matches site/js/snake.js's shape: state()/describe()/render()/
// candidates()/step()/selfTest().

// The state reports observed distances; the model evaluates the distance limits in RULES.
// Candidates gate lane-change legality only, never whether to brake for an observed signal.
// RULES order = precedence order = candidate order.
export const RULES = {
  stop: "applies when a pedestrian is crossing the car's lane within 15 m ahead",
  brake: 'applies when the traffic light within 25 m ahead is red',
  'change lane right': 'applies when the car must move to the right lane',
  accelerate: 'applies when the road ahead is clear for 50 m', // speeds up to the limit; no-op at it
  'change lane left': 'applies when the car must move to the left lane',
  follow: 'applies when the car cannot pass an obstruction ahead and must slow down',
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
const PED_RANGE = 120; // m - observation range, not a stop trigger
const LIGHT_RANGE = 120; // m - observation range, not a brake trigger
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

  // A lane is "clear" with no vehicle OR cone within LANE_CLEAR_GAP, ahead or behind - a cone
  // sitting in the target lane blocks a lane change into it exactly like a car would.
  _laneClear(lane) {
    const near = (pos) => Math.abs(pos - this.ego.position) <= LANE_CLEAR_GAP;
    return !this.traffic.some((c) => c.lane === lane && near(c.position)) && !this.cones.some((c) => c.lane === lane && near(c.pos));
  }

  // Nearest thing occupying the car's own lane ahead - a traffic car (its real speed) or a cone
  // (a static, zero-speed obstruction) - closest one wins. Unifying these means a cone runs
  // through the exact same pass/follow/collision logic a stalled car already does: nothing the
  // ego can drive through without either changing lane or slowing to its pace.
  _obstacles() {
    const e = this.ego;
    const traffic = this.traffic.map((c) => ({ position: c.position, speed: c.speed, lane: c.lane, isCone: false }));
    const cones = this.cones.map((c) => ({ position: c.pos, speed: 0, lane: c.lane, isCone: true }));
    return [...traffic, ...cones]
      .filter((o) => o.lane === e.lane && o.position > e.position && o.position - e.position <= LEAD_RANGE)
      .sort((a, b) => a.position - b.position);
  }

  _lead() {
    return this._obstacles()[0] ?? null;
  }

  // The situations describe() reports and candidates() gates on - one boolean per rule condition.
  situations() {
    const e = this.ego;
    const lead = this._lead();
    const light = this.lights.find((l) => l.pos >= e.position && l.pos - e.position <= LIGHT_RANGE);
    const ped = this.pedestrians.find((p) => p.lane === e.lane && p.pos >= e.position && p.pos - e.position <= PED_RANGE);
    const exitIn = DEST_POS - e.position;
    const exitNear = exitIn > 0 && exitIn <= EXIT_RANGE;
    const leftClear = e.lane > 1 && this._laneClear(e.lane - 1);
    const rightClear = e.lane < LANES && this._laneClear(e.lane + 1);
    // "blocked": something slower than the limit sits close enough ahead that the car must pass
    // it or slow to its pace - checked against SPEED_LIMIT, not the car's own current speed, so a
    // car that has already matched a stopped obstruction's pace (speed 0 == lead's speed 0) keeps
    // re-evaluating whether it can now pass, instead of latching into "hold speed" forever the
    // moment the gap stops closing (the bug behind "stops dead instead of overtaking").
    const blocked = Boolean(lead && lead.speed < SPEED_LIMIT && lead.position - e.position <= CLOSING_RANGE);
    return {
      lead,
      light,
      ped,
      exitIn,
      red: Boolean(light && light.state === 'red'),
      blocked,
      // Passing used to be left-only, so an obstruction with just the right lane free left the
      // model nothing legal to pick and the car sat behind it. The engine picks the side (left
      // first, the fast lane) and the sentence names it, so the rule stays one condition.
      passSide: blocked && !exitNear ? (leftClear ? 'left' : rightClear ? 'right' : null) : null,
      canPass: blocked && !exitNear && (leftClear || rightClear),
      exitNear,
      mustExit: exitNear && e.lane < LANES,
      roadClear: !lead,
      belowLimit: e.speed < SPEED_LIMIT,
      leftClear,
      rightClear,
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
    // `placed: true` distinguishes a visitor-placed hazard from a procedurally spawned one so
    // the renderer can draw a marker ring only under what the visitor actually did.
    if (kind === 'pedestrian') this.pedestrians.push({ pos, lane, ticksLeft: 20, placed: true }); // longer-lived
    // than a spawned pedestrian's 4 ticks - a visitor-placed one should stick around to watch.
    else if (kind === 'cone') this.cones.push({ pos, lane, placed: true });
    else if (kind === 'car') this.traffic.push({ lane, position: pos, speed: 0, cruise: 0, placed: true });
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
    } else if (action === 'change lane left') e.lane -= 1;
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

    // Collision/occupancy test: a cone is exactly as solid as a traffic car here. This is the
    // backstop for "driving through a cone" - the rules above should stop the car first, but if
    // a model ignores them the run ends instead of sailing through for free. Checked as a swept
    // interval (prevPos..position, widened by COLLISION_GAP), not just the final gap: a static
    // cone plus a fast, accelerating car can cover more than COLLISION_GAP in one 0.5 s tick and
    // step clean over an end-of-tick-only check without either position ever landing within it.
    const hitLane = (lane, objPos) =>
      lane === this.ego.lane && objPos + COLLISION_GAP >= prevPos && objPos - COLLISION_GAP <= this.ego.position;
    const hitTraffic = this.traffic.some((c) => hitLane(c.lane, c.position));
    const hitCone = this.cones.some((c) => hitLane(c.lane, c.pos));
    if (hitTraffic || hitCone) {
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

  // Keep observation order stable; include distances even outside the action ranges.
  describe() {
    const e = this.ego;
    const sit = this.situations();
    const sentences = [
      `The car is in lane ${e.lane} of ${LANES} at ${e.speed} km/h, ${sit.belowLimit ? 'below' : 'at'} the speed limit of ${SPEED_LIMIT} km/h.`,
    ];
    if (sit.ped) sentences.push(`A pedestrian is crossing the car's lane ${Math.round(sit.ped.pos - e.position)} m ahead.`);
    if (sit.red) sentences.push(`The traffic light ${Math.round(sit.light.pos - e.position)} m ahead is red.`);
    // Unambiguous, agent-subject, precomputed: a slower vehicle and a static cone both "obstruct"
    // the lane, so both feed the same change-lane-left/follow pair - one sentence when passing is
    // legal, a different one when it isn't, never the same wording for both rules (that ambiguity
    // was why the model sat behind slower traffic instead of overtaking).
    if (sit.lead) {
      const obstacle = sit.lead.isCone ? 'cone' : 'vehicle';
      if (sit.canPass) sentences.push(`The car must move to the ${sit.passSide} lane to pass the ${obstacle} ahead.`);
      else if (sit.blocked) sentences.push(`The car cannot pass the ${obstacle} ahead and must slow down.`);
      else if (sit.lead.isCone) sentences.push(`A cone is ${Math.round(sit.lead.position - e.position)} m ahead.`);
      else sentences.push(`The lead vehicle is ${Math.round(sit.lead.position - e.position)} m ahead at ${sit.lead.speed} km/h.`);
    }
    if (sit.mustExit) sentences.push(`The car must move to the right lane to take the exit in ${Math.round(sit.exitIn)} m.`);
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
    stop: Boolean(sit.ped && sit.ped.pos - engine.ego.position <= 15),
    brake: sit.red && sit.light.pos - engine.ego.position <= 25,
    'change lane right': sit.mustExit || (sit.canPass && sit.passSide === 'right'),
    accelerate: sit.roadClear,
    'change lane left': sit.canPass && sit.passSide === 'left',
    follow: sit.blocked,
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
  d2.lights = [{ pos: 1025, state: 'red', timer: 5 }];
  d2.pedestrians = [];
  console.assert(
    d2.describe() ===
      'The car is in lane 2 of 3 at 45 km/h, below the speed limit of 60 km/h. The traffic light 25 m ahead is red. The car must move to the left lane to pass the vehicle ahead. The lane to the left is clear.',
    'describe() lists only the situations that apply, in the fixed order'
  );
  console.assert(!d2.candidates().includes('change lane right'), 'lane change blocked by a car within 15 m');
  console.assert(d2.candidates().includes('change lane left'), 'clear lane offered');
  console.assert(greedyPolicy(d2) === 'brake', 'red light outranks the lane change');
  d2.lights = [];
  console.assert(greedyPolicy(d2) === 'change lane left', 'slower vehicle ahead with a clear left lane -> change lane left (overtake)');
  d2.traffic.push({ lane: 1, position: 1005, speed: 40 });
  console.assert(greedyPolicy(d2) === 'follow', 'no clear lane -> follow');
  console.assert(d2.describe().includes('The car cannot pass the vehicle ahead and must slow down.'), 'unpassable obstruction renders the follow sentence, not the ambiguous "closing" one');
  d2.step('follow');
  console.assert(d2.ego.speed === 30, 'follow matches the lead speed');

  // exit + clear road + pedestrian
  const d5 = new Drive({ seed: 1 });
  d5.ego = { lane: 2, speed: 40, position: 1300 };
  d5.traffic = [];
  d5.lights = [];
  d5.pedestrians = [];
  console.assert(d5.describe() === 'The car is in lane 2 of 3 at 40 km/h, below the speed limit of 60 km/h. The car must move to the right lane to take the exit in 200 m. The road ahead is clear for 50 m. The lane to the left is clear. The lane to the right is clear.', 'exit sentence precedes the clear-road sentence');
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

  // visitor-placed hazards flow through the same pipeline as spawned ones, including the
  // pass/follow logic - a cone is a static, zero-speed obstruction, not a special case.
  const d6 = new Drive({ seed: 1 });
  d6.ego = { lane: 2, speed: 40, position: 500 };
  d6.traffic = [];
  d6.lights = [];
  d6.pedestrians = [];
  d6.cones = [];
  console.assert(!d6.placeHazard('cone', 2, 480), 'hazard behind the car is rejected');
  console.assert(!d6.placeHazard('cone', 4, 520), 'hazard off-road is rejected');
  console.assert(d6.placeHazard('cone', 2, 515), 'on-road hazard ahead of the car is accepted');
  console.assert(d6.cones[0].placed === true, "placed hazards are flagged so the renderer can ring-mark only what the visitor placed");
  console.assert(d6.describe().includes('The car must move to the left lane to pass the cone ahead.'), 'a passable cone names the side to move to');
  console.assert(greedyPolicy(d6) === 'change lane left', 'cone with a clear left lane -> change lane left (not driven through)');
  d6.traffic.push({ lane: 1, position: 510, speed: 40, cruise: 40 }); // block the left lane
  console.assert(d6.describe().includes('The car must move to the right lane to pass the cone ahead.'), 'with the left lane blocked the car passes on the right');
  console.assert(greedyPolicy(d6) === 'change lane right', 'an obstruction with only the right lane free is passed on the right, not sat behind');
  d6.traffic.push({ lane: 3, position: 512, speed: 40, cruise: 40 }); // block the right lane too
  console.assert(d6.describe().includes('The car cannot pass the cone ahead and must slow down.'), 'an unpassable cone renders the follow sentence');
  console.assert(greedyPolicy(d6) === 'follow', 'cone with no clear lane -> follow (slows toward a stop, never a pass-through)');
  console.assert(d6.placeHazard('pedestrian', 3, 510), 'pedestrian hazard accepted ahead in a different lane');
  console.assert(d6.placeHazard('car', 3, 530), 'stopped-car hazard accepted (reuses the traffic array)');
  console.assert(d6.traffic.some((c) => c.speed === 0 && c.placed), 'stopped-car hazard has zero speed and is flagged as placed');

  // a cone the car cannot pass must stop it, not let it cruise through at speed. Both neighbour
  // lanes stay permanently blocked by stationary (speed 0) cars near where the ego will stop, so
  // the test isn't accidentally passing because the blockers drove off.
  const d7 = new Drive({ seed: 1 });
  d7.ego = { lane: 2, speed: 50, position: 0 };
  d7.traffic = [
    { lane: 1, position: 10, speed: 0, cruise: 0 },
    { lane: 3, position: 30, speed: 0, cruise: 0 },
  ];
  d7.lights = [];
  d7.pedestrians = [];
  d7.cones = [{ lane: 2, pos: 40 }];
  for (let i = 0; i < 40 && !d7.done; i++) d7.step(greedyPolicy(d7));
  console.assert(d7.ego.position < 40, "following the rule list, the car never reaches an unavoidable cone's position");
  console.assert(d7.ego.speed === 0, 'the car stops for the cone instead of driving through it');

  // the same cone, driven through on purpose (model ignoring the rules): the collision/occupancy
  // test has to catch it exactly like it catches a traffic car - no free pass-through.
  const d8 = new Drive({ seed: 1 });
  d8.ego = { lane: 2, speed: 50, position: 0 };
  d8.traffic = [];
  d8.lights = [];
  d8.pedestrians = [];
  d8.cones = [{ lane: 2, pos: 20 }];
  for (let i = 0; i < 10 && !d8.done; i++) d8.step('accelerate');
  console.assert(d8.done && d8.collisions === 1, 'driving through a cone registers a collision, same as hitting traffic');

  // overtaking resumes once the lane clears, even after the car has already matched a blocker's
  // speed to zero - the bug that made it "stop dead" instead of ever passing.
  const d9 = new Drive({ seed: 1 });
  d9.ego = { lane: 2, speed: 0, position: 0 };
  d9.traffic = [
    { lane: 2, position: 10, speed: 0, cruise: 0 }, // stopped dead ahead, ego already matched its pace
    { lane: 1, position: 5, speed: 40, cruise: 40 }, // left lane currently blocked
    { lane: 3, position: 6, speed: 40, cruise: 40 }, // and the right one too, so neither side is free
  ];
  d9.lights = [];
  d9.pedestrians = [];
  d9.cones = [];
  console.assert(d9.situations().blocked && !d9.situations().canPass, 'boxed in: blocked but cannot pass yet');
  console.assert(greedyPolicy(d9) === 'follow', 'follow (hold at the blocker\'s pace) while boxed in');
  d9.traffic[1].position = 200; // the blocking car in lane 1 moves well clear
  console.assert(d9.situations().canPass && d9.situations().passSide === 'left', 'once the left lane clears the car re-evaluates and can pass, even at matched speed 0');
  console.assert(greedyPolicy(d9) === 'change lane left', 'passing resumes instead of sitting behind the blocker forever');

  const crossing = new Drive({ seed: 1 });
  crossing.traffic = [];
  crossing.lights = [];
  crossing.placeHazard('pedestrian', 2, 120);
  console.assert(crossing.describe().includes("A pedestrian is crossing the car's lane 120 m ahead."), 'distant placed pedestrian retains its distance');
  console.assert(greedyPolicy(crossing) === 'accelerate', 'reference policy approaches a distant pedestrian');
  crossing.ego.position = 108;
  console.assert(greedyPolicy(crossing) === 'stop', 'reference policy stops near a pedestrian');

  // Distance is an observation, never an override of the selected action.
  const approach = new Drive({ seed: 1 });
  approach.ego = { lane: 2, speed: 60, position: 0 };
  approach.traffic = [];
  approach.lights = [{ pos: 120, state: 'red', timer: 100 }];
  approach.rng = () => 1; // no random pedestrians in this signal check
  console.assert(approach.describe().includes('The traffic light 120 m ahead is red.'), 'distant signal is reported with distance');
  console.assert(greedyPolicy(approach) === 'accelerate', 'reference policy approaches a distant red');
  for (let i = 0; i < 20 && approach.ego.speed > 0; i++) approach.step(greedyPolicy(approach));
  console.assert(approach.ego.speed === 0 && 120 - approach.ego.position > 0 && 120 - approach.ego.position < 15, 'reference policy stops near the signal');
  console.assert(approach.violations === 0, 'stops before the signal');
  approach.ego = { lane: 2, speed: 60, position: 0 };
  approach.lights = [{ pos: 20, state: 'red', timer: 100 }];
  approach.step('accelerate');
  console.assert(approach.ego.speed === 60, 'engine does not veto an action at a red light');

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
