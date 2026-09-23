#!/usr/bin/env node
// Records 200 live model decisions against real id DOOM (site/games/doom/index.html) via the
// FastAPI server, for render-realdoom.js's honest static-mode tally ("last recorded model
// run: ..."). Drives window.Doom through page.evaluate and calls the server directly from
// Node (not from the page) so this works with a plain static file server and no api.js
// changes. Needs, in two other terminals:
//   python3 -m http.server 8788 -d site
//   uv run uvicorn server:app --port 8787
// Usage: node scripts/record_realdoom.mjs
import fs from 'node:fs';
import { chromium } from '/Users/guynachshon/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs';
import { candidatesFor, describeDoom, resolveIntent, scriptedPolicy, keyPress } from '../site/js/games/realdoom-logic.js';
import { QUESTION } from '../site/js/games/doom.js';

const PAGE = 'http://localhost:8788/games/doom/index.html';
const SERVER = 'http://localhost:8787';
const N = 200;

async function decide(state, queries) {
  const res = await fetch(`${SERVER}/api/decide`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'typical-small', state, queries }),
  });
  if (!res.ok) throw new Error(`decide failed: ${res.status} ${await res.text()}`);
  return res.json();
}

async function waitForLevel(page, timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const ok = await page.evaluate(() => Boolean(window.Doom?.ready && window.Doom.state().in_level)).catch(() => false);
    if (ok) return true;
    await new Promise((r) => setTimeout(r, 200));
  }
  return false;
}

async function main() {
  const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] });
  const page = await browser.newPage();
  page.on('pageerror', (err) => console.error('[pageerror]', err));
  await page.goto(PAGE);
  if (!(await waitForLevel(page))) throw new Error('DOOM never reached in_level within 15s');

  const decisions = [];
  let shots = 0;
  let rule_agreement = 0;
  let kills = 0;
  let health_end = 100;

  for (let tic = 0; tic < N; tic++) {
    const state = await page.evaluate(() => window.Doom.state());
    if (!state.in_level) break;
    const candidates = candidatesFor(state);
    if (candidates.length === 0) break;
    const desc = describeDoom(state);

    let move;
    let probs;
    let ms;
    if (candidates.length === 1) {
      [move, probs, ms] = [candidates[0], { [candidates[0]]: 1 }, 0];
    } else {
      const t0 = Date.now();
      const res = await decide(desc, [{ type: 'choice', question: QUESTION, labels: candidates }]);
      ms = Date.now() - t0;
      const r = res.results[0];
      probs = r.probs;
      move = candidates.reduce((best, m) => ((r.probs[m] ?? 0) > (r.probs[best] ?? 0) ? m : best), candidates[0]);
    }

    const gold = scriptedPolicy(state); // the rule list applied literally
    if (move === gold) rule_agreement += 1;
    const action = resolveIntent(state, move); // the model aims; the engine presses the key
    const [key, hold] = keyPress(state, move, action);
    if (action === 'shoot') shots += 1;
    const killsBefore = state.kills ?? 0;
    await page.evaluate(({ key, hold }) => (typeof hold === 'object' ? window.Doom.turnBy(key, hold.deg) : window.Doom.press(key, hold)), { key, hold });
    const after = await page.evaluate(() => window.Doom.state());
    if ((after.kills ?? 0) > killsBefore) kills += after.kills - killsBefore;
    health_end = after.health ?? health_end;

    decisions.push({ tic, desc, candidates, probs: candidates.map((m) => probs[m] ?? 0), move, gold, action, ms, x: Math.round(state.x), y: Math.round(state.y) });
    console.log(`[${tic}] ${move} -> ${action} (${ms}ms) — ${desc.slice(0, 90)}`);

    if (!after.in_level) break; // died / demo kicked back to menu
  }

  const count = (k) => decisions.filter((d) => d.move === k).length;
  const summary = { decisions: decisions.length, rule_agreement, shots, kills, health_end, by_label: Object.fromEntries(['retreat', 'shoot', 'turn left', 'turn right', 'explore'].map((k) => [k, count(k)])) };
  fs.writeFileSync(new URL('../site/data/replays/realdoom.json', import.meta.url), JSON.stringify({ decisions, summary }));
  console.log('STATS_JSON', JSON.stringify(summary));
  await browser.close();
}

main();
