// Real id Software DOOM (Chocolate Doom -> WASM, games/doom/index.html) mounted in an iframe -
// same-origin, so we read window.Doom straight off iframe.contentWindow. This is the "does it
// actually work on the real game" companion to render-doom.js's own ASCII/Three.js arena:
// demo-spec-v2.md #5 found the model says "shoot" 29/30 times regardless of state on that
// engine, and this card lets the same failure play out against id's actual E1M1.
import { TOKENS, mountChrome, paintDecision, watchVisibility, createTicker, createHumanOverride, bindKeys, modelPolicy, loadJSON, scoreboardLine } from './loop.js';
import { candidatesFor, describeDoom, scriptedPolicy, resolveIntent, resetNav, keyPress, replayIndex, recordedDecision } from './realdoom-logic.js';
import { QUESTION } from './doom.js';

const TICK_MS = 400;
const WASM_TIMEOUT_MS = 10000;
const KEYMAP = {
  ArrowUp: 'move forward', ArrowDown: 'move back', ArrowLeft: 'turn left', ArrowRight: 'turn right', ' ': 'shoot',
  w: 'move forward', s: 'move back', a: 'turn left', d: 'turn right',
};

// [key, ms] -> a timed press; [key, { deg }] -> a closed-loop turn (games/doom/index.html).
// A shoot decision keeps firing during its bounded aim turn, instead of alternating aim/fire
// across model ticks and repeatedly passing the target.
function applyMove(doom, spec, fireWhileTurning = false) {
  if (!spec || !doom) return Promise.resolve();
  if (fireWhileTurning && typeof spec[1] === 'object' && doom.turnAndFire) return doom.turnAndFire(spec[0], spec[1].deg);
  return typeof spec[1] === 'object' ? doom.turnBy(spec[0], spec[1].deg) : doom.press(spec[0], spec[1]);
}

// Polls iframe.contentWindow.Doom until the level is actually running (Doom.ready flips as
// soon as the wasm boots, but newGame() fires ~1.5s later inside the harness) or times out.
function waitForDoom(iframe, timeoutMs) {
  return new Promise((resolve) => {
    const start = performance.now();
    const poll = () => {
      let ok = false;
      try {
        const w = iframe.contentWindow;
        ok = !!(w?.Doom?.ready && w.Doom.state().in_level);
      } catch {
        // iframe document not ready yet
      }
      if (ok) return resolve(true);
      if (performance.now() - start > timeoutMs) return resolve(false);
      setTimeout(poll, 200);
    };
    poll();
  });
}

// Hides the harness's debug <pre> and stretches its fixed 640x400 canvas to fill the card
// (image-rendering:pixelated keeps it crisp) - we own none of games/doom/index.html, so this
// is done from the outside via the (same-origin) iframe document rather than editing it.
function primeIframeStyles(iframe) {
  try {
    const doc = iframe.contentDocument;
    const style = doc.createElement('style');
    style.textContent = `
      html, body { margin:0; height:100%; background:#000; overflow:hidden; }
      #state { display:none; }
      canvas { width:100% !important; height:100% !important; image-rendering:pixelated; display:block; }
    `;
    doc.head.appendChild(style);
  } catch {
    // cross-origin (shouldn't happen, same-origin harness) - canvas stays at native 640x400
  }
}

// WASM didn't come up in time - fall back to the untouched Three.js arena and say so on top
// of it. render-doom.js owns its own HUD (repainted every frame), so the "why" banner has to
// live as a sibling overlay rather than text handed into its mount().
async function mountFallback(el, opts, reason) {
  const { mount: mountThree } = await import('./render-doom.js');
  const handle = await mountThree(el, opts);
  const banner = document.createElement('div');
  banner.textContent = `real DOOM offline — ${reason} — showing the arena fallback`;
  banner.style.cssText = `position:absolute; top:26px; left:12px; right:12px; z-index:5; font-size:9px; letter-spacing:.08em; text-transform:uppercase; color:${TOKENS.paper}; background:rgba(0,0,0,.65); padding:4px 6px;`;
  el.querySelector('.gc-stage')?.appendChild(banner);
  return handle;
}

export async function mount(el, { decide, mode } = {}) {
  const refs = mountChrome(el, { label: 'DOOM · E1M1 · SHAREWARE 1.9' });
  refs.canvas.remove(); // real DOOM draws inside the iframe below, not the shared chrome canvas

  const iframe = document.createElement('iframe');
  iframe.src = 'games/doom/index.html';
  iframe.title = 'DOOM (Chocolate Doom / WASM)';
  iframe.setAttribute('allow', 'autoplay');
  iframe.style.cssText = 'position:absolute; inset:0; width:100%; height:100%; border:0; display:block;';
  refs.stage.appendChild(iframe);

  const replayDoc = await loadJSON('data/replays/realdoom.json').catch(() => null);
  const replay = replayIndex(replayDoc);

  const ready = await waitForDoom(iframe, WASM_TIMEOUT_MS);
  if (!ready) return mountFallback(el, { decide, mode }, 'WASM module did not reach the level within 10s');
  primeIframeStyles(iframe);

  const doom = iframe.contentWindow.Doom;
  let policyName = 'model';
  let busy = false; // ponytail: one in-flight tick - key presses (up to 350ms) can outlast the 400ms interval
  let currState = doom.state();
  let lastDecision = { candidates: candidatesFor(currState), probs: {}, p_null: null, sentence: describeDoom(currState) };
  let shots = 0;
  let decisions = 0;
  let agreed = 0;
  let kills = currState.kills ?? 0;
  const human = createHumanOverride(3000);
  let killScore = { you: 0, model: 0 }; // "you vs model" HUD line, reset on restart

  const foot = refs.sentence.parentElement;
  const tallyEl = document.createElement('div');
  tallyEl.className = 'gc-sentence';
  tallyEl.style.marginTop = '4px';
  const offlineEl = document.createElement('div');
  offlineEl.className = 'gc-sentence';
  offlineEl.style.marginTop = '4px';
  foot.append(tallyEl, offlineEl);

  function restart() {
    shots = 0;
    decisions = 0;
    agreed = 0;
    resetNav();
    doom.releaseAll();
    doom.newGame(3);
    kills = 0;
    killScore = { you: 0, model: 0 };
  }

  function paintHUD(state) {
    refs.hud.innerHTML = '';
    const l1 = document.createElement('div');
    l1.textContent = 'DOOM · E1M1 · SHAREWARE 1.9';
    const l2 = document.createElement('div');
    l2.textContent = `HEALTH ${state.health} · AMMO ${state.ammo} · KILLS ${state.kills ?? 0}/${state.total_kills ?? 0}`;
    refs.hud.append(l1, l2);
    const scoreLine = scoreboardLine(killScore, 'kills');
    if (scoreLine) refs.hud.appendChild(document.createElement('div')).textContent = scoreLine;
  }

  function paintTally() {
    tallyEl.textContent = `model decisions ${decisions} · matched the rule list ${agreed} · shots ${shots} · kills ${kills}`;
  }

  function paintOffline() {
    if (mode() === 'live') {
      offlineEl.textContent = '';
      return;
    }
    const s = replayDoc?.summary;
    const recorded = s ? ` Last recorded model run: ${s.decisions} decisions · ${s.rule_agreement} matched the rule list · shots ${s.shots} · kills ${s.kills} · health end ${s.health_end}.` : '';
    offlineEl.textContent = `model offline — replaying recorded model decisions; unrecorded states use the scripted policy.${recorded}`;
  }

  async function tick() {
    if (busy) return;
    busy = true;
    try {
      currState = doom.state();
      if (!currState.in_level) return;
      const legal = candidatesFor(currState);
      if (legal.length === 0) return;
      const sentence = describeDoom(currState);
      const offline = mode() !== 'live';

      const humanTurn = human.active();
      let move;
      let decision;
      const gold = scriptedPolicy(currState); // the rule list applied literally
      if (humanTurn) {
        move = human.move; // a raw key action; resolveIntent() passes it through
        decision = { candidates: legal, probs: { [move]: 1 }, p_null: 0, sentence };
      } else if (policyName === 'scripted') {
        move = gold;
        decision = { candidates: legal, probs: { [move]: 1 }, p_null: 0, sentence };
      } else if (offline) {
        const hit = recordedDecision(replay, sentence, legal);
        move = hit?.move ?? gold;
        decision = { candidates: legal, probs: hit?.probs ?? { [move]: 1 }, p_null: hit?.p_null ?? 0, sentence };
        if (hit) {
          decisions += 1;
          if (move === gold) agreed += 1;
        }
      } else {
        const r = await modelPolicy({ decide, engine: { candidates: () => legal, describe: () => sentence }, question: QUESTION });
        move = r?.move ?? gold;
        decision = { candidates: legal, probs: r?.probs ?? {}, p_null: r?.p_null ?? null, sentence };
        decisions += 1;
        if (move === gold) agreed += 1;
      }
      lastDecision = decision;

      const action = resolveIntent(currState, move); // the model aims; the engine presses the key
      const killsBefore = currState.kills ?? 0;
      const ammoBefore = currState.ammo ?? 0;
      await applyMove(doom, keyPress(currState, move, action), move === 'shoot' && action.startsWith('turn '));
      const after = doom.state();
      shots += Math.max(0, ammoBefore - (after.ammo ?? ammoBefore));
      if ((after.kills ?? 0) > killsBefore) {
        const delta = after.kills - killsBefore;
        kills += delta;
        if (humanTurn) killScore.you += delta;
        else killScore.model += delta;
      }

      paintHUD(after);
      paintDecision(refs, lastDecision);
      paintTally();
      paintOffline();
    } finally {
      busy = false;
    }
  }

  paintHUD(currState);
  paintDecision(refs, lastDecision);
  paintTally();
  paintOffline();

  const ticker = createTicker(TICK_MS, tick);
  ticker.start();
  const stopWatch = watchVisibility(el, (v) => (v ? ticker.resume() : ticker.pause()));
  const unbindKeys = bindKeys(el, KEYMAP, (action) => human.set(action));

  refs.policyBtn.textContent = 'Model';
  refs.policyBtn.classList.add('gc-on');
  refs.policyBtn.addEventListener('click', () => {
    policyName = policyName === 'model' ? 'scripted' : 'model';
    refs.policyBtn.textContent = policyName === 'model' ? 'Model' : 'Scripted';
    refs.policyBtn.classList.toggle('gc-on', policyName === 'model');
  });
  refs.restartBtn.addEventListener('click', restart);

  return {
    stop() {
      ticker.stop();
      stopWatch();
      unbindKeys();
      try {
        doom.releaseAll();
      } catch {
        // iframe already torn down
      }
    },
    restart,
    setPolicy(name) {
      policyName = name === 'scripted' ? 'scripted' : 'model';
      refs.policyBtn.textContent = policyName === 'model' ? 'Model' : 'Scripted';
      refs.policyBtn.classList.toggle('gc-on', policyName === 'model');
    },
  };
}
