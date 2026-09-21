// Playwright check: game boots into E1M1, state() has health>0, forward press moves x/y.
import { chromium } from '/Users/guynachshon/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs';
const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 700, height: 900 } });
page.on('console', m => { if (m.type() === 'error') console.log('[console.error]', m.text().slice(0, 300)); });
page.on('pageerror', e => console.log('[pageerror]', String(e).slice(0, 300)));
await page.goto('http://localhost:8788/games/doom/index.html');
try {
  await page.waitForFunction(() => window.Doom && Doom.ready && Doom.state().in_level && !Doom.state().demo, null, { timeout: 90000 });
} catch (e) {
  console.log('did not reach level; last state:', await page.evaluate(() => JSON.stringify(Doom.state())));
  await page.screenshot({ path: '/Users/guynachshon/conductor/workspaces/typical/abuja/.context/shots/doom-wasm.png' });
  await browser.close(); process.exit(1);
}
await page.waitForTimeout(1500);
const s0 = await page.evaluate(() => Doom.state());
console.log('state0', JSON.stringify(s0));
await page.evaluate(() => Doom.press('forward', 500));
await page.waitForTimeout(800);
const s1 = await page.evaluate(() => Doom.state());
console.log('state1', JSON.stringify(s1));
await page.evaluate(() => Doom.press('right', 400));
await page.waitForTimeout(700);
const s2 = await page.evaluate(() => Doom.state());
console.log('state2 angle', s2.angle, 'monsters', JSON.stringify(s2.monsters));
await page.screenshot({ path: '/Users/guynachshon/conductor/workspaces/typical/abuja/.context/shots/doom-wasm.png' });
const moved = Math.hypot(s1.x - s0.x, s1.y - s0.y);
console.log('health', s0.health, 'moved', moved.toFixed(1), 'turned', (s2.angle - s1.angle).toFixed(1));
if (!(s0.health > 0 && moved > 1)) { console.log('FAIL'); process.exit(1); }
console.log('PASS');
await browser.close();
