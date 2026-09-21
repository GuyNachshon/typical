// Three.js first-person renderer over js/games/doom.js's pure grid-arena engine. Flat-shaded,
// texture-free: ink walls with vellum EdgesGeometry outlines, bone floor, ink ceiling, billboard
// "imp" sprites (ink-on-paper canvas texture, 3 poses) for enemies, paper/bone pickups, a
// DOM weapon sprite + muzzle flash + damage vignette. This is a Doom-LIKE built for the page
// over our own grid engine, not id's DOOM - the HUD label says so.
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js';
import { Doom, greedyPolicy, QUESTION } from './doom.js';
import { TOKENS, mountChrome, paintDecision, watchVisibility, createTicker, createHumanOverride, bindKeys, modelPolicy, replayFrame, loadJSON } from './loop.js';

const TICK_MS = 300;
const CELL = 2;
const WALL_H = 2.6;
const EYE_H = 1.3;
// Mirrors js/games/doom.js's internal HEADINGS (not exported) - only the unit deltas are
// needed here, to turn a facing index into a forward vector for the camera.
const HEADINGS = [
  [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1],
];

const KEYMAP = {
  ArrowUp: 'move forward', ArrowDown: 'move back', ArrowLeft: 'turn left', ArrowRight: 'turn right', ' ': 'shoot',
  w: 'move forward', s: 'move back', a: 'turn left', d: 'turn right',
};

function lerp(a, b, t) {
  return a + (b - a) * t;
}
function cellCenter(gx, gy) {
  return { x: gx * CELL + CELL / 2, z: gy * CELL + CELL / 2 };
}

// Small ink-on-paper "imp" plaque, three poses: idle (legs together), walk (legs apart), dead
// (fallen, squashed). Drawn once per pose into a canvas texture - no external art assets.
function impTexture(pose) {
  const c = document.createElement('canvas');
  c.width = 64;
  c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = TOKENS.paper;
  g.fillRect(0, 0, 64, 64);
  g.strokeStyle = TOKENS.ink;
  g.lineWidth = 2;
  g.strokeRect(3, 3, 58, 58);
  g.fillStyle = TOKENS.ink;
  if (pose === 'dead') {
    g.beginPath();
    g.ellipse(32, 46, 22, 8, 0, 0, Math.PI * 2);
    g.fill();
    g.beginPath();
    g.arc(14, 44, 7, 0, Math.PI * 2);
    g.fill();
  } else {
    g.beginPath();
    g.arc(32, 16, 9, 0, Math.PI * 2); // head
    g.fill();
    g.beginPath();
    g.moveTo(20, 24);
    g.lineTo(44, 24);
    g.lineTo(40, 46);
    g.lineTo(24, 46);
    g.closePath();
    g.fill(); // torso
    const spread = pose === 'walk' ? 10 : 4;
    g.fillRect(24 - spread * 0.5, 46, 6, 16); // left leg
    g.fillRect(34 + spread * 0.5 - 6, 46, 6, 16); // right leg
    g.fillRect(14, 26, 6, 16); // left arm
    g.fillRect(44, 26, 6, 16); // right arm
  }
  const tex = new THREE.CanvasTexture(c);
  tex.magFilter = THREE.NearestFilter;
  return tex;
}

function buildArena(grid, w, h) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(TOKENS.ink);
  scene.add(new THREE.AmbientLight(TOKENS.bone, 0.55));
  const lamp = new THREE.PointLight(TOKENS.paper, 0.9, 30);
  lamp.position.set(0, WALL_H - 0.3, 0);
  scene.add(lamp);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(w * CELL, h * CELL),
    new THREE.MeshLambertMaterial({ color: TOKENS.bone })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.set((w * CELL) / 2, 0, (h * CELL) / 2);
  scene.add(floor);

  const ceil = new THREE.Mesh(
    new THREE.PlaneGeometry(w * CELL, h * CELL),
    new THREE.MeshLambertMaterial({ color: TOKENS.ink })
  );
  ceil.rotation.x = Math.PI / 2;
  ceil.position.set((w * CELL) / 2, WALL_H, (h * CELL) / 2);
  scene.add(ceil);

  const wallGeo = new THREE.BoxGeometry(CELL, WALL_H, CELL);
  const wallMat = new THREE.MeshLambertMaterial({ color: TOKENS.ink, flatShading: true });
  const edgeGeo = new THREE.EdgesGeometry(wallGeo);
  const edgeMat = new THREE.LineBasicMaterial({ color: TOKENS.vellum });
  const solid = new Set(['#', 'O']);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!solid.has(grid[y][x])) continue;
      const { x: wx, z: wz } = cellCenter(x, y);
      const mesh = new THREE.Mesh(wallGeo, wallMat);
      mesh.position.set(wx, WALL_H / 2, wz);
      mesh.add(new THREE.LineSegments(edgeGeo, edgeMat));
      scene.add(mesh);
    }
  }
  return scene;
}

function buildPickupMesh(type) {
  const color = type === 'ammo' ? TOKENS.bone : TOKENS.paper;
  return new THREE.Mesh(new THREE.OctahedronGeometry(0.22, 0), new THREE.MeshLambertMaterial({ color }));
}

export async function mount(el, { decide, mode } = {}) {
  const refs = mountChrome(el, { label: 'Arena · our engine' });
  const canvas = refs.canvas;

  // data/replays/doom.json is {frames, summary} (scripts/record_games.mjs) - replayFrame()
  // wants the bare frames array.
  const replay = ((await loadJSON('data/replays/doom.json').catch(() => null)) ?? {}).frames ?? [];

  const probe = new Doom({ seed: 1 }); // grid is fixed regardless of seed - read once for the static geometry
  const W = probe.grid[0].length;
  const H = probe.grid.length;
  const scene = buildArena(probe.grid, W, H);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  const camera = new THREE.PerspectiveCamera(78, 1, 0.05, 100);

  const textures = { idle: impTexture('idle'), walk: impTexture('walk'), dead: impTexture('dead') };
  const enemySprites = probe.enemies.map(() => {
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: textures.idle }));
    spr.scale.set(1.3, 1.3, 1);
    scene.add(spr);
    return spr;
  });

  const pickupMeshes = new Map(); // items index -> mesh
  probe.items.forEach((item, i) => {
    const mesh = buildPickupMesh(item.type);
    const { x, z } = cellCenter(item.x, item.y);
    mesh.position.set(x, 0.5, z);
    scene.add(mesh);
    pickupMeshes.set(i, mesh);
  });

  let engine = new Doom({ seed: Math.floor(Math.random() * 1e9) });
  let policyName = 'model';
  let prevState = engine.state();
  let currState = engine.state();
  let lastTickAt = performance.now();
  let replayIndex = 0;
  let lastDecision = { candidates: engine.candidates(), probs: {}, p_null: null, sentence: engine.describe() };
  const human = createHumanOverride(3000);
  let pendingRestart = false;
  let muzzleUntil = 0;
  let kickUntil = 0;
  let vignetteUntil = 0;

  function restart() {
    pendingRestart = false;
    engine = new Doom({ seed: Math.floor(Math.random() * 1e9) });
    prevState = engine.state();
    currState = engine.state();
    replayIndex = 0;
    lastTickAt = performance.now();
    lastDecision = { candidates: engine.candidates(), probs: {}, p_null: null, sentence: engine.describe() };
  }

  function applyEffects(move, prevHealth, currHealth) {
    const now = performance.now();
    if (move === 'shoot') {
      muzzleUntil = now + 90;
      kickUntil = now + 120;
    }
    if (currHealth < prevHealth) vignetteUntil = now + 400;
  }

  async function tick() {
    if (pendingRestart) return;
    if (policyName === 'model' && mode() !== 'live') {
      const f = replayFrame(replay, replayIndex);
      if (!f) return;
      replayIndex = f.nextIndex;
      prevState = currState;
      currState = f.state;
      lastTickAt = performance.now();
      lastDecision = { candidates: f.candidates, probs: f.probs, p_null: f.p_null, sentence: f.desc };
      applyEffects(f.move, prevState.player.health, currState.player.health);
      if (currState.dead) {
        pendingRestart = true;
        setTimeout(restart, 900);
      }
      return;
    }

    if (engine.dead) return;
    const legal = engine.candidates();
    if (legal.length === 0) {
      currState = engine.state();
      pendingRestart = true;
      setTimeout(restart, 900);
      return;
    }

    let move;
    let decision;
    if (human.active()) {
      move = legal.includes(human.move) ? human.move : legal[0];
      decision = { candidates: legal, probs: { [move]: 1 }, p_null: 0, sentence: engine.describe() };
    } else if (policyName === 'scripted') {
      move = greedyPolicy(engine);
      decision = { candidates: legal, probs: { [move]: 1 }, p_null: 0, sentence: engine.describe() };
    } else {
      const r = await modelPolicy({ decide, engine, question: QUESTION });
      move = r?.move ?? greedyPolicy(engine);
      decision = { candidates: legal, probs: r?.probs ?? {}, p_null: r?.p_null ?? null, sentence: engine.describe() };
    }

    prevState = engine.state();
    engine.step(move);
    currState = engine.state();
    lastTickAt = performance.now();
    lastDecision = decision;
    applyEffects(move, prevState.player.health, currState.player.health);
    if (engine.dead) {
      pendingRestart = true;
      setTimeout(restart, 900);
    }
  }

  function resize() {
    const rect = refs.stage.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(refs.stage);
  resize();

  function draw() {
    const t = Math.max(0, Math.min(1, (performance.now() - lastTickAt) / TICK_MS));
    const pp = prevState.player;
    const cp = currState.player;
    const p0 = cellCenter(pp.x, pp.y);
    const p1 = cellCenter(cp.x, cp.y);
    const camX = lerp(p0.x, p1.x, t);
    const camZ = lerp(p0.z, p1.z, t);
    const [d0x, d0y] = HEADINGS[pp.h];
    const [d1x, d1y] = HEADINGS[cp.h];
    let fx = lerp(d0x, d1x, t);
    let fz = lerp(d0y, d1y, t);
    const mag = Math.hypot(fx, fz) || 1;
    fx /= mag;
    fz /= mag;
    camera.position.set(camX, EYE_H, camZ);
    camera.lookAt(camX + fx, EYE_H, camZ + fz);

    currState.enemies.forEach((e, i) => {
      const spr = enemySprites[i];
      const prevE = prevState.enemies[i] ?? e;
      const c0 = cellCenter(prevE.x, prevE.y);
      const c1 = cellCenter(e.x, e.y);
      spr.position.set(lerp(c0.x, c1.x, t), 0.75, lerp(c0.z, c1.z, t));
      const moved = prevE.x !== e.x || prevE.y !== e.y;
      const pose = !e.alive ? 'dead' : moved ? 'walk' : 'idle';
      spr.material.map = textures[pose];
      spr.visible = true;
    });

    currState.items.forEach((item, i) => {
      const mesh = pickupMeshes.get(i);
      if (!mesh) return;
      mesh.visible = !item.taken;
      if (!item.taken) mesh.position.y = 0.5 + Math.sin(performance.now() / 300 + i) * 0.08;
    });

    renderer.render(scene, camera);
  }

  function frame() {
    draw();
    refs.hud.innerHTML = '';
    const l1 = document.createElement('div');
    l1.textContent = 'Arena · our engine';
    const l2 = document.createElement('div');
    l2.textContent = `health ${currState.player.health} · ammo ${currState.player.ammo}`;
    refs.hud.append(l1, l2);
    paintDecision(refs, lastDecision);

    const now = performance.now();
    weapon.style.transform = now < kickUntil ? 'translate(-50%, 6px)' : 'translate(-50%, 0)';
    muzzle.style.opacity = now < muzzleUntil ? '1' : '0';
    const vignetteOn = now < vignetteUntil;
    vignette.style.opacity = vignetteOn ? '0.5' : '0';

    rafId = requestAnimationFrame(frame);
  }

  // DOM overlays: weapon sprite (ink silhouette, kicks on fire), muzzle flash, damage vignette.
  // No gradients (design system rule) - the vignette is a flat bone frame of 4 bars, not a
  // radial fade.
  const weapon = document.createElement('div');
  weapon.style.cssText = `position:absolute; left:50%; bottom:0; width:120px; height:70px; transform:translate(-50%,0); transition:transform .08s ease-out; pointer-events:none; z-index:2; background:${TOKENS.ink}; clip-path:polygon(40% 0%, 60% 0%, 78% 45%, 100% 55%, 100% 100%, 0% 100%, 0% 55%, 22% 45%);`;
  const muzzle = document.createElement('div');
  muzzle.style.cssText = `position:absolute; left:50%; bottom:52px; width:40px; height:40px; transform:translateX(-50%); background:${TOKENS.paper}; opacity:0; pointer-events:none; z-index:2; clip-path:polygon(50% 0%,61% 35%,98% 35%,68% 57%,79% 100%,50% 76%,21% 100%,32% 57%,2% 35%,39% 35%);`;
  const vignette = document.createElement('div');
  vignette.style.cssText = `position:absolute; inset:0; pointer-events:none; z-index:2; opacity:0; transition:opacity .12s ease-out; box-shadow:none; border:14px solid ${TOKENS.bone};`;
  refs.stage.append(vignette, weapon, muzzle);

  let rafId = requestAnimationFrame(frame);

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
      cancelAnimationFrame(rafId);
      ticker.stop();
      stopWatch();
      unbindKeys();
    },
    restart,
    setPolicy(name) {
      policyName = name === 'scripted' ? 'scripted' : 'model';
      refs.policyBtn.textContent = policyName === 'model' ? 'Model' : 'Scripted';
      refs.policyBtn.classList.toggle('gc-on', policyName === 'model');
    },
  };
}
