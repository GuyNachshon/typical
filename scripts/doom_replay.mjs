#!/usr/bin/env node
// A reproducible DOOM run: same start, same inputs on the same tics, same trace.
//
// Our existing recorder (scripts/record_realdoom.mjs) drives the engine on wall clock -- press(key,
// ms) and a turnBy() that loops on performance.now(). DOOM advances in tics, so a 180ms hold is
// however many tics the machine happened to fit into 180ms, and two runs of the same decisions
// diverge. That is why the 200-decision run on the site is a recording nobody else can regenerate.
//
// DOOM itself is deterministic: P_Random reads a fixed 256-entry table whose index M_ClearRandom
// resets at level start, and the simulation advances a tic at a time. It is the reason .lmp demo
// files work at all. So reproducibility does not need a different engine -- it needs us to stop
// measuring in milliseconds.
//
// Here every action is held for an exact number of tics and every decision happens on an exact tic
// boundary. Run it twice and the checksum matches, which is the claim the site could not make.
//
//   node scripts/doom_replay.mjs --decisions 60 --policy scripted
//   node scripts/doom_replay.mjs --decisions 60 --policy model     (needs the server on :8787)
//
// Needs a static server on :8788 (python3 -m http.server 8788 -d site).
//
// STATE OF THIS: the reproducibility works and is verified -- two model-driven runs of 60 decisions
// produce byte-identical traces (checksum 3f46f45cfe670ade), and the scripted policy likewise. What
// is NOT done is the action layer. render-realdoom.js drives the game through keyPress()/turnBy(),
// which turn the player by DEGREES in a closed loop on performance.now(); that is what lets the
// live card aim into an 8-degree crosshair, and it is wall-clock by construction. This harness
// bypasses it with a flat action->key map so that every input edge can be pinned to a tic, and the
// cost is that the run is not smart: it walks the E1M1 spawn corridor and stops, 220 decisions of
// "explore" and no kills, because it cannot aim or work a door.
//
// Porting turnBy to turn-by-tics is the remaining work, and it changes behaviour rather than just
// timing, so it needs re-tuning against the live card. Until then this file proves the method and
// does not pretend to be a run worth publishing.

import fs from 'node:fs';
import crypto from 'node:crypto';
import { chromium } from '/Users/guynachshon/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs';
import { candidatesFor, describeDoom, scriptedPolicy } from '../site/js/games/realdoom-logic.js';
import { QUESTION } from '../site/js/games/doom.js';

const PAGE = 'http://localhost:8788/games/doom/index.html';
const SERVER = 'http://localhost:8787';

// The run is defined entirely by these. Change one and you get a different, still reproducible run.
const CONFIG = {
  skill: 3,          // E1M1, "Hurt me plenty" -- the menu's default
  settleTics: 40,    // let the level finish spawning before the first decision
  holdTics: 6,       // every action is held for exactly this many tics
  // One decision every this many tics. It has to be comfortably longer than a decision takes to
  // come back, because the engine keeps running while the model thinks: at ~35 tics/s this is
  // ~850ms against ~300ms of inference. Too small and the run misses its own boundaries, which is
  // the failure the assertion below exists to make loud instead of silent.
  cadenceTics: 30,
  // The press starts on a fixed tic too. Holding as soon as the model answers put the key down at
  // boundary + however long inference took, so two runs pressed on different tics and the player
  // ended up fractionally apart even when every decision tic matched. Every input edge is pinned.
  pressOffsetTics: 20,
};

const arg = (k, d) => {
  const i = process.argv.indexOf(`--${k}`);
  return i > 0 ? process.argv[i + 1] : d;
};

// Hold a key for an exact number of tics. The engine free-runs, so we watch its own counter rather
// than a clock: the input goes down on one tic boundary and up on another, the same two every time.
async function holdTics(page, key, tics) {
  await page.evaluate(async ({ key, tics }) => {
    const tic = () => window.Doom.state().tic;
    const start = tic();
    window.Doom.key(key, true);
    while (tic() - start < tics) await new Promise((r) => setTimeout(r, 2));
    window.Doom.key(key, false);
  }, { key, tics });
}

async function waitTic(page, target) {
  await page.waitForFunction((t) => window.Doom.state().tic >= t, target, { timeout: 30000 });
}

// What the run is pinned on. Position, facing, health, ammo and kills at every decision: if the
// simulation diverges anywhere, one of these moves.
function checksum(trace) {
  const h = crypto.createHash('sha256');
  trace.forEach((d) => h.update(`${d.tic}|${d.x}|${d.y}|${d.angle}|${d.health}|${d.ammo}|${d.kills}|${d.action}\n`));
  return h.digest('hex').slice(0, 16);
}

async function decideWithModel(state, queries) {
  const res = await fetch(`${SERVER}/api/decide`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state, queries }),
  });
  if (!res.ok) throw new Error(`decide failed: ${res.status}`);
  return res.json();
}

async function run({ decisions, policy }) {
  const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] });
  const page = await browser.newPage();
  await page.goto(PAGE, { waitUntil: 'load' });
  await page.waitForFunction(() => window.Doom && window.Doom.ready, { timeout: 60000 });
  await page.evaluate((skill) => window.Doom.newGame(skill), CONFIG.skill);
  await page.waitForFunction(() => window.Doom.state().in_level, { timeout: 30000 });

  const t0 = await page.evaluate(() => window.Doom.state().tic);
  await waitTic(page, t0 + CONFIG.settleTics);

  // Boundaries come from a fixed origin, not from the tic we happened to observe last time. The
  // first version added the cadence to the observed tic, so any lateness compounded and two runs
  // of the same policy drifted apart -- decision 1 landed on tic 65 in one run and 53 in another.
  const origin = await page.evaluate(() => window.Doom.state().tic);
  const trace = [];
  let late = 0;
  for (let i = 0; i < decisions; i += 1) {
    const boundary = origin + i * CONFIG.cadenceTics;
    await waitTic(page, boundary);
    const s = await page.evaluate(() => window.Doom.state());
    // arriving after the boundary means the previous decision overran its slot: the trace from here
    // is no longer reproducible, so say so rather than writing a file that claims it is
    if (s.tic > boundary + 1) {
      late += 1;
      if (late > 0) throw new Error(`missed tic boundary at decision ${i}: arrived at ${s.tic}, wanted ${boundary}. Raise cadenceTics.`);
    }
    if (!s.in_level) break;
    const cands = candidatesFor(s);
    let action;
    if (policy === 'model') {
      const out = await decideWithModel(describeDoom(s), [{ type: 'choice', question: QUESTION, labels: cands }]);
      action = out.results[0].argmax;
    } else {
      action = scriptedPolicy(s);
    }
    if (!cands.includes(action)) action = cands[0];
    trace.push({ i, tic: s.tic, x: s.x, y: s.y, angle: s.angle, health: s.health, ammo: s.ammo, kills: s.kills, action });
    const key = { 'move forward': 'forward', 'move back': 'back', 'turn left': 'left', 'turn right': 'right', shoot: 'fire', use: 'use', retreat: 'back', explore: 'forward' }[action] || 'forward';
    const pressAt = boundary + CONFIG.pressOffsetTics;
    await waitTic(page, pressAt);
    const at = await page.evaluate(() => window.Doom.state().tic);
    if (at > pressAt + 1) throw new Error(`decision ${i} took too long: press wanted tic ${pressAt}, arrived ${at}. Raise pressOffsetTics.`);
    await holdTics(page, key, CONFIG.holdTics);
  }
  await browser.close();
  return trace;
}

// The one thing worth asserting without a browser: boundaries come from the origin, so lateness
// in one slot cannot move any later slot.
export function boundariesFrom(origin, n, cadence) {
  return Array.from({ length: n }, (_, i) => origin + i * cadence);
}

if (process.argv.includes('--demo')) {
  const b = boundariesFrom(41, 4, 30);
  console.assert(b.join() === '41,71,101,131', 'boundaries are absolute, not cumulative');
  console.assert(b[3] - b[0] === 90, 'and the span is exactly three cadences');
  console.assert(CONFIG.pressOffsetTics + CONFIG.holdTics < CONFIG.cadenceTics, 'press and hold fit inside one slot');
  console.log('doom_replay.mjs self-test OK');
  process.exit(0);
}

const decisions = Number(arg('decisions', 60));
const policy = arg('policy', 'scripted');
const trace = await run({ decisions, policy });
const sum = checksum(trace);
const out = { config: CONFIG, policy, decisions: trace.length, checksum: sum, trace };
const path = arg('out', `runs/doom_${policy}.json`);
fs.mkdirSync(path.replace(/\/[^/]+$/, ''), { recursive: true });
fs.writeFileSync(path, `${JSON.stringify(out, null, 1)}\n`);
console.log(`${policy}: ${trace.length} decisions, checksum ${sum}, kills ${trace.at(-1)?.kills ?? 0} -> ${path}`);
