#!/usr/bin/env node
// Records a seeded, deterministic playthrough of Snake, Doom or Drive driven by the live
// server, one tick per model decision. Used by site/js/games/render-{snake,doom,drive}.js's
// static-mode "Model" playback: each frame carries `state` (engine.state() snapshot), not
// ASCII, so the renderer can draw the real board/arena/road instead of replaying text.
// Requires server.py running on :8787 (uv run uvicorn server:app --port 8787).
//
// Usage: node scripts/record_games.mjs --game snake|doom|drive
import fs from 'node:fs';
import { Snake } from '../site/js/snake.js';
import { Doom } from '../site/js/games/doom.js';
import { Drive } from '../site/js/games/drive.js';
import { QUESTION as DOOM_QUESTION } from '../site/js/demos/doom.js';
import { QUESTION as DRIVE_QUESTION } from '../site/js/demos/drive.js';

const SERVER = 'http://localhost:8787';
const SEED = 7;

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const game = process.argv.includes('--game') ? process.argv[process.argv.indexOf('--game') + 1] : null;
if (!['snake', 'doom', 'drive'].includes(game)) throw new Error('--game must be snake, doom or drive');

async function decide(state, queries) {
  const res = await fetch(`${SERVER}/api/decide`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'typical-small', state, queries }),
  });
  if (!res.ok) throw new Error(`decide failed: ${res.status} ${await res.text()}`);
  return res.json();
}

// Shared record loop: `legal(engine)` returns the candidate labels, `isDone(engine)` reports
// game-over, `step` advances one tick. A single-candidate choice is forced (server.py requires
// >=2 labels) - skip the network round trip.
async function recordLoop({ engine, maxTicks, legal, isDone, question }) {
  const frames = [];
  for (let tick = 0; tick < maxTicks; tick++) {
    const candidates = legal(engine);
    if (candidates.length === 0) break;

    const desc = engine.describe();
    let move, probs, p_null, ms;
    if (candidates.length === 1) {
      [move, probs, p_null, ms] = [candidates[0], { [candidates[0]]: 1 }, 0, 0];
    } else {
      const t0 = Date.now();
      const res = await decide(desc, [{ type: 'choice', question, labels: candidates }]);
      ms = Date.now() - t0;
      const r = res.results[0];
      probs = r.probs;
      p_null = r.p_null;
      move = candidates.reduce((best, m) => ((r.probs[m] ?? 0) > (r.probs[best] ?? 0) ? m : best), candidates[0]);
    }

    frames.push({ tick, state: engine.state(), desc, candidates, probs: candidates.map((m) => probs[m] ?? 0), p_null, move, ms });
    engine.step(move);
    if (isDone(engine)) {
      frames.push({ tick: tick + 1, state: engine.state(), desc: engine.describe(), candidates: [], probs: [], p_null: 0, move: null, ms: 0 });
      break;
    }
  }
  return frames;
}

async function recordSnake() {
  Math.random = mulberry32(SEED); // Snake._placeFood() uses the global RNG
  const presets = JSON.parse(fs.readFileSync(new URL('../site/data/presets.json', import.meta.url)));
  const { w, h } = presets.snake.board;
  const engine = new Snake({ w, h });
  const manhattan = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

  let foodEaten = 0;
  let foodwardCount = 0;
  const frames = await recordLoop({
    engine,
    maxTicks: 300,
    legal: (e) => e.safeMoves(),
    isDone: (e) => e.dead,
    question: presets.snake.question,
  });

  for (const f of frames) {
    if (!f.move) continue;
    const { head, food } = f.state;
    const before = manhattan(head, food);
    const D = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
    const [dx, dy] = D[f.move];
    if (manhattan({ x: head.x + dx, y: head.y + dy }, food) < before) foodwardCount++;
  }
  // food eaten = number of ticks where the recorded body length grew vs the previous frame
  for (let i = 1; i < frames.length; i++) {
    if (frames[i].state.body.length > frames[i - 1].state.body.length) foodEaten++;
  }

  const decided = frames.filter((f) => f.move);
  const summary = {
    game: 'snake',
    seed: SEED,
    ticks: frames.length,
    food_eaten: foodEaten,
    foodward_fraction: decided.length ? foodwardCount / decided.length : 0,
    mean_p_null: decided.reduce((s, f) => s + f.p_null, 0) / (decided.length || 1),
    mean_ms: decided.reduce((s, f) => s + f.ms, 0) / (decided.length || 1),
  };
  fs.writeFileSync(new URL('../site/data/replays/snake.json', import.meta.url), JSON.stringify({ frames, summary }));
  console.log(`[snake] ticks=${frames.length} food_eaten=${foodEaten} foodward_fraction=${summary.foodward_fraction.toFixed(3)}`);
  console.log(`[snake] mean p_null=${summary.mean_p_null.toFixed(3)} mean ms=${summary.mean_ms.toFixed(1)}`);
  console.log(`STATS_JSON ${JSON.stringify(summary)}`);
}

async function recordDoom() {
  const engine = new Doom({ seed: SEED });
  let shots = 0;
  let shotsCrosshair = 0;
  let kills = 0;

  const frames = await recordLoop({
    engine,
    maxTicks: 240,
    legal: (e) => e.candidates(),
    isDone: (e) => e.dead,
    question: DOOM_QUESTION,
  });

  for (const f of frames) {
    if (f.move !== 'shoot') continue;
    shots++;
    if (f.desc.includes('in the crosshair')) shotsCrosshair++;
  }
  for (let i = 1; i < frames.length; i++) {
    if (frames[i].state.score > frames[i - 1].state.score) kills++;
  }

  const decided = frames.filter((f) => f.move);
  const summary = {
    game: 'doom',
    seed: SEED,
    ticks: frames.length,
    kills,
    shots,
    shots_with_enemy_in_crosshair: shotsCrosshair,
    mean_p_null: decided.reduce((s, f) => s + f.p_null, 0) / (decided.length || 1),
    mean_ms: decided.reduce((s, f) => s + f.ms, 0) / (decided.length || 1),
  };
  fs.writeFileSync(new URL('../site/data/replays/doom.json', import.meta.url), JSON.stringify({ frames, summary }));
  console.log(`[doom] ticks=${frames.length} kills=${kills} shots=${shots} shots_with_crosshair=${shotsCrosshair}`);
  console.log(`[doom] mean p_null=${summary.mean_p_null.toFixed(3)} mean ms=${summary.mean_ms.toFixed(1)}`);
  console.log(`STATS_JSON ${JSON.stringify(summary)}`);
}

async function recordDrive() {
  const engine = new Drive({ seed: SEED });
  let redTotal = 0;
  let redRespected = 0;
  let pedTotal = 0;
  let pedStopped = 0;
  let laneChanges = 0;

  const frames = await recordLoop({
    engine,
    maxTicks: 240,
    legal: (e) => e.candidates(),
    isDone: (e) => e.done,
    question: DRIVE_QUESTION,
  });

  for (const f of frames) {
    if (!f.move) continue;
    const e = f.state.ego;
    const redLight = f.state.lights.find((l) => l.pos >= e.position && l.pos - e.position <= 60 && l.state === 'red');
    if (redLight) {
      redTotal++;
      if (f.move === 'brake' || f.move === 'stop') redRespected++;
    }
    const pedNear = f.state.pedestrians.some((p) => p.lane === e.lane && Math.abs(p.pos - e.position) <= 20);
    if (pedNear) {
      pedTotal++;
      if (f.move === 'stop') pedStopped++;
    }
    if (f.move === 'change lane left' || f.move === 'change lane right') laneChanges++;
  }

  const decided = frames.filter((f) => f.move);
  const last = frames[frames.length - 1].state;
  const summary = {
    game: 'drive',
    seed: SEED,
    ticks: frames.length,
    distance_m: Math.round(last.distance),
    red_lights_run: last.violations,
    red_lights_total: redTotal,
    red_lights_respected: redRespected,
    pedestrian_stops: pedStopped,
    pedestrians_total: pedTotal,
    lane_changes: laneChanges,
    collisions: last.collisions,
    arrived: last.arrived,
    mean_p_null: decided.reduce((s, f) => s + f.p_null, 0) / (decided.length || 1),
    mean_ms: decided.reduce((s, f) => s + f.ms, 0) / (decided.length || 1),
  };
  fs.writeFileSync(new URL('../site/data/replays/drive.json', import.meta.url), JSON.stringify({ frames, summary }));
  console.log(`[drive] ticks=${frames.length} distance=${summary.distance_m}m red_lights_run=${last.violations} lane_changes=${laneChanges} collisions=${last.collisions}`);
  console.log(`[drive] red respected=${redRespected}/${redTotal} pedestrian stops=${pedStopped}/${pedTotal}`);
  console.log(`[drive] mean p_null=${summary.mean_p_null.toFixed(3)} mean ms=${summary.mean_ms.toFixed(1)}`);
  console.log(`STATS_JSON ${JSON.stringify(summary)}`);
}

({ snake: recordSnake, doom: recordDoom, drive: recordDrive })[game]();
