#!/usr/bin/env node
// Records a seeded, deterministic playthrough of Snake driven by the live server, one tick
// per model decision. Used by js/snake-ui.js's replayPolicy in static deployments (no local
// server) to show a real recorded game instead of nothing.
// Requires server.py running on :8787 (uv run uvicorn server:app --port 8787).
//
// --variant plain (default): describe() with no options -> site/data/snake-replay.json
// --variant facts: describe({distanceFacts:true}), the spec's pre-registered fallback (b)
//   -> site/data/snake-replay-facts.json
import fs from 'node:fs';
import { Snake } from '../site/js/snake.js';

// ponytail: Snake._placeFood() calls the global Math.random() - overriding the global with a
// seeded PRNG is the simplest way to get a reproducible game without threading a custom RNG
// through the engine (which nothing else needs).
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
Math.random = mulberry32(7);

const SERVER = 'http://127.0.0.1:8787';
const DELTA = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
const MAX_TICKS = 300;

const variantArg = process.argv.includes('--variant') ? process.argv[process.argv.indexOf('--variant') + 1] : 'plain';
if (!['plain', 'facts'].includes(variantArg)) throw new Error(`--variant must be plain or facts, got ${variantArg}`);
const distanceFacts = variantArg === 'facts';
const outFile = variantArg === 'facts' ? 'snake-replay-facts.json' : 'snake-replay.json';

const presets = JSON.parse(fs.readFileSync(new URL('../site/data/presets.json', import.meta.url)));
const { w, h } = presets.snake.board;
const question = presets.snake.question;

async function decide(state, queries) {
  const res = await fetch(`${SERVER}/api/decide`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'typical-small', state, queries }),
  });
  if (!res.ok) throw new Error(`decide failed: ${res.status} ${await res.text()}`);
  return res.json();
}

const manhattan = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

async function main() {
  const engine = new Snake({ w, h });
  const frames = [];
  let foodEaten = 0;
  let argmaxCloserCount = 0;
  let safeMovesTotal = 0;
  let safeMovesCloser = 0;
  let pNullSum = 0;
  let msSum = 0;
  let tick = 0;

  for (; tick < MAX_TICKS; tick++) {
    const safe = engine.safeMoves();
    if (safe.length === 0) break; // boxed in - no lookahead

    const desc = engine.describe({ distanceFacts });
    const { head, food } = engine.state();
    const distBefore = manhattan(head, food);

    // A single-candidate choice is trivial and the server requires >=2 labels (see
    // server.py's DecideRequest validation) - skip the call, the outcome is forced.
    let move, probs, p_null, ms;
    if (safe.length === 1) {
      [move, probs, p_null, ms] = [safe[0], { [safe[0]]: 1 }, 0, 0];
    } else {
      const t0 = Date.now();
      const res = await decide(desc, [{ type: 'choice', question, labels: safe }]);
      ms = Date.now() - t0;
      const r = res.results[0];
      probs = r.probs;
      p_null = r.p_null;
      move = safe.reduce((best, m) => ((r.probs[m] ?? 0) > (r.probs[best] ?? 0) ? m : best), safe[0]);
    }

    safe.forEach((m) => {
      safeMovesTotal++;
      const [dx, dy] = DELTA[m];
      if (manhattan({ x: head.x + dx, y: head.y + dy }, food) < distBefore) safeMovesCloser++;
    });
    const [mdx, mdy] = DELTA[move];
    if (manhattan({ x: head.x + mdx, y: head.y + mdy }, food) < distBefore) argmaxCloserCount++;

    frames.push({
      tick,
      ascii: engine.render(),
      desc,
      safe,
      probs: safe.map((m) => probs[m] ?? 0),
      p_null,
      move,
      ms,
    });
    pNullSum += p_null;
    msSum += ms;

    const scoreBefore = engine.score;
    engine.step(move);
    if (engine.score > scoreBefore) foodEaten++;
    if (engine.dead) {
      tick++;
      break;
    }
  }

  const outPath = new URL(`../site/data/${outFile}`, import.meta.url);
  fs.writeFileSync(outPath, JSON.stringify(frames));

  const n = frames.length;
  const stats = {
    variant: variantArg,
    ticks: tick,
    food_eaten: foodEaten,
    final_length: engine.body.length,
    argmax_foodward_fraction: argmaxCloserCount / n,
    safe_move_foodward_fraction: safeMovesCloser / safeMovesTotal,
    mean_p_null: pNullSum / n,
    mean_ms: msSum / n,
  };
  console.log(`[variant=${variantArg}]`);
  console.log(`ticks survived: ${tick} (${n} recorded decisions)`);
  console.log(`food eaten: ${foodEaten}`);
  console.log(`final length: ${engine.body.length}`);
  console.log(`fraction of argmax moves that reduced Manhattan distance to food: ${stats.argmax_foodward_fraction.toFixed(3)}`);
  console.log(`fraction of all offered safe moves that would have reduced it: ${stats.safe_move_foodward_fraction.toFixed(3)}`);
  console.log(`mean p_null: ${stats.mean_p_null.toFixed(3)}`);
  console.log(`mean ms: ${stats.mean_ms.toFixed(1)}`);
  console.log(`wrote ${outPath.pathname} (${(fs.statSync(outPath.pathname).size / 1024).toFixed(0)} KB)`);
  // Machine-readable line for scripting the plain-vs-facts comparison.
  console.log(`STATS_JSON ${JSON.stringify(stats)}`);
}

main();
