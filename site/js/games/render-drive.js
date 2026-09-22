// Three.js orthographic 3/4 renderer over js/games/drive.js's pure engine. Flat-shaded
// low-poly: putty ground, graphite road, bone lane dashes/traffic, ink ego wedge with paper
// windows, black-post traffic lights (lit = paper lamp), bone pedestrian capsules, an exit ramp
// splitting off right near the destination.
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js';
import { Drive, greedyPolicy, QUESTION, ROAD_LENGTH, DEST_POS } from './drive.js';
import { TOKENS, mountChrome, paintDecision, watchVisibility, createTicker, createHumanOverride, bindKeys, modelPolicy, replayFrame, loadJSON } from './loop.js';

const TICK_MS = 400;
// Mirrors js/games/drive.js's internal constants (not exported - small fixed numbers, not
// worth threading through a second module for).
const LANES = 3;
const LANE_W = 3.2;

// Follow camera: shallower than straight-down (CAM_HEIGHT small next to CAM_BACK+CAM_AHEAD) so
// more road depth fits in the same ortho frustum (see resize()'s viewSize comment) - lane
// dashes and traffic still read fine at this angle, see resize().
const CAM_HEIGHT = 9;
const CAM_BACK = 14; // m behind the ego the camera sits
const CAM_AHEAD = 22; // m ahead of the ego the camera looks at (frustum centre)

const KEYMAP = {
  ArrowLeft: 'change lane left', ArrowRight: 'change lane right',
  ArrowUp: 'accelerate', ArrowDown: 'brake', ' ': 'stop',
  a: 'change lane left', d: 'change lane right', w: 'accelerate', s: 'brake',
};

function laneToX(lane) {
  return (lane - (LANES + 1) / 2) * LANE_W;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

// Manual convex hexagonal-footprint wedge (pointed front, flat back) - the low-poly "car" the
// design brief asks for without pulling in an extra geometry example module.
function wedgeGeometry(hw, hl, height) {
  const bottom = [
    [-hw, 0, -hl], [hw, 0, -hl], [hw, 0, 0.15 * hl],
    [0.35 * hw, 0, hl], [-0.35 * hw, 0, hl], [-hw, 0, 0.15 * hl],
  ];
  const top = bottom.map(([x, , z]) => [x, height, z]);
  const positions = [];
  const pushTri = (a, b, c) => positions.push(...a, ...b, ...c);
  // bottom fan (reverse winding so the normal faces down) + top fan
  for (let i = 1; i < bottom.length - 1; i++) pushTri(bottom[0], bottom[i + 1], bottom[i]);
  for (let i = 1; i < top.length - 1; i++) pushTri(top[0], top[i], top[i + 1]);
  // sides
  for (let i = 0; i < bottom.length; i++) {
    const j = (i + 1) % bottom.length;
    pushTri(bottom[i], bottom[j], top[j]);
    pushTri(bottom[i], top[j], top[i]);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.computeVertexNormals();
  return geo;
}

// Radial vignette + horizon band, drawn once into a CanvasTexture and set as scene.background -
// a background render pass is always behind every scene object (z-order guaranteed, not a DOM
// overlay), so it can never sit over the HUD, which is a separate absolutely-positioned element.
function buildSkyTexture() {
  const size = 512;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size * 0.42, size * 0.05, size / 2, size * 0.42, size * 0.75);
  g.addColorStop(0, '#141414');
  g.addColorStop(1, TOKENS.putty);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// Soft dark disc (contact shadow / cheap AO) reused across every ego/traffic instance.
function buildShadowTexture() {
  const size = 64;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, 'rgba(0,0,0,0.45)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
}

function buildShadow(shadowTex, w, d) {
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(w, d),
    new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false })
  );
  mesh.rotation.x = -Math.PI / 2;
  return mesh;
}

function buildScene() {
  const scene = new THREE.Scene();
  scene.background = buildSkyTexture();

  scene.add(new THREE.AmbientLight(TOKENS.bone, 0.75));
  const sun = new THREE.DirectionalLight(TOKENS.paper, 0.7);
  sun.position.set(-20, 40, -10);
  scene.add(sun);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(60, ROAD_LENGTH + 100),
    new THREE.MeshLambertMaterial({ color: TOKENS.putty })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(0, 0, ROAD_LENGTH / 2);
  scene.add(ground);

  // shoulder: a strip either side of the road, between road and putty ground
  const roadHalfW = (LANE_W * LANES) / 2;
  const shoulderW = 1.4;
  const shoulderMat = new THREE.MeshLambertMaterial({ color: TOKENS.vellum });
  for (const side of [-1, 1]) {
    const shoulder = new THREE.Mesh(new THREE.PlaneGeometry(shoulderW, ROAD_LENGTH + 100), shoulderMat);
    shoulder.rotation.x = -Math.PI / 2;
    shoulder.position.set(side * (roadHalfW + shoulderW / 2), 0.008, ROAD_LENGTH / 2);
    scene.add(shoulder);
  }

  const road = new THREE.Mesh(
    new THREE.PlaneGeometry(LANE_W * LANES, ROAD_LENGTH + 100),
    new THREE.MeshLambertMaterial({ color: TOKENS.graphite })
  );
  road.rotation.x = -Math.PI / 2;
  road.position.set(0, 0.01, ROAD_LENGTH / 2);
  scene.add(road);

  // solid edge lines (road/shoulder boundary) - ponytail: a one-way 3-lane road has no true
  // centre line (that's for opposing traffic); the internal lane boundaries stay dashed below.
  const edgeMat = new THREE.MeshLambertMaterial({ color: TOKENS.bone });
  for (const x of [-roadHalfW, roadHalfW]) {
    const line = new THREE.Mesh(new THREE.PlaneGeometry(0.12, ROAD_LENGTH + 100), edgeMat);
    line.rotation.x = -Math.PI / 2;
    line.position.set(x, 0.015, ROAD_LENGTH / 2);
    scene.add(line);
  }

  // exit ramp: a widening wedge peeling off the right shoulder near DEST_POS
  const rightEdge = laneToX(LANES) + LANE_W / 2;
  const rz0 = DEST_POS - 140;
  const rz1 = DEST_POS + 40;
  const rz2 = DEST_POS + 95;
  const rampVerts = new Float32Array([
    rightEdge, 0.015, rz0, rightEdge, 0.015, rz1, rightEdge + 6, 0.015, rz2,
    rightEdge, 0.015, rz0, rightEdge + 6, 0.015, rz2, rightEdge + 2.2, 0.015, rz0,
  ]);
  const rampGeo = new THREE.BufferGeometry();
  rampGeo.setAttribute('position', new THREE.BufferAttribute(rampVerts, 3));
  rampGeo.computeVertexNormals();
  scene.add(new THREE.Mesh(rampGeo, new THREE.MeshLambertMaterial({ color: TOKENS.graphite, side: THREE.DoubleSide })));

  // lane dashes on the 2 internal lane boundaries
  const dashGeo = new THREE.BoxGeometry(0.25, 0.03, 1.6);
  const dashMat = new THREE.MeshLambertMaterial({ color: TOKENS.bone });
  const boundaries = [laneToX(1) + LANE_W / 2, laneToX(2) + LANE_W / 2];
  const dashSpacing = 6;
  const dashCount = boundaries.length * Math.ceil(ROAD_LENGTH / dashSpacing);
  const dashes = new THREE.InstancedMesh(dashGeo, dashMat, dashCount);
  let di = 0;
  const m4 = new THREE.Matrix4();
  for (const x of boundaries) {
    for (let z = 0; z < ROAD_LENGTH; z += dashSpacing) {
      m4.makeTranslation(x, 0.02, z);
      dashes.setMatrixAt(di++, m4);
    }
  }
  dashes.instanceMatrix.needsUpdate = true;
  scene.add(dashes);

  return { scene, sun, road };
}

// Returns { group, brakeLights } - brakeLights is the material shared by both rear lamps, toggled
// between an unlit ink dot and a lit paper one whenever the engine's speed drops between ticks.
function buildEgo() {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    wedgeGeometry(0.9, 1.9, 0.7),
    new THREE.MeshLambertMaterial({ color: TOKENS.ink, flatShading: true })
  );
  group.add(body);
  const windowMat = new THREE.MeshBasicMaterial({ color: TOKENS.paper });
  const windshield = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 0.9), windowMat);
  windshield.rotation.x = -Math.PI / 2;
  windshield.position.set(0, 0.71, 0.3);
  group.add(windshield);

  const brakeMat = new THREE.MeshBasicMaterial({ color: TOKENS.ink });
  const lampGeo = new THREE.BoxGeometry(0.2, 0.12, 0.06);
  for (const x of [-0.6, 0.6]) {
    const lamp = new THREE.Mesh(lampGeo, brakeMat);
    lamp.position.set(x, 0.4, -1.88);
    group.add(lamp);
  }
  return { group, brakeMat };
}

function buildTrafficCar() {
  return new THREE.Mesh(
    new THREE.BoxGeometry(1.7, 1.1, 3.4),
    new THREE.MeshLambertMaterial({ color: TOKENS.bone, flatShading: true })
  );
}

function buildCone() {
  return new THREE.Mesh(
    new THREE.ConeGeometry(0.35, 0.8, 6),
    new THREE.MeshLambertMaterial({ color: TOKENS.ink, flatShading: true })
  );
}

function buildPedestrian() {
  return new THREE.Mesh(
    new THREE.CapsuleGeometry(0.28, 0.7, 4, 8),
    new THREE.MeshLambertMaterial({ color: TOKENS.bone })
  );
}

function buildLight(pos) {
  const group = new THREE.Group();
  const post = new THREE.Mesh(
    new THREE.CylinderGeometry(0.08, 0.08, 3.2, 8),
    new THREE.MeshLambertMaterial({ color: TOKENS.ink })
  );
  post.position.y = 1.6;
  const lamp = new THREE.Mesh(
    new THREE.BoxGeometry(0.35, 0.35, 0.35),
    new THREE.MeshBasicMaterial({ color: TOKENS.ink })
  );
  lamp.position.y = 3.3;
  group.add(post, lamp);
  const x = laneToX(LANES) + LANE_W / 2 + 0.6;
  group.position.set(x, 0, pos);
  return { group, lamp };
}

// Injected once (scoped to .gc-drive, this module's own root class - never leaks into the
// snake/doom cards which share loop.js's .gc-root) for the one thing no existing class covers:
// a crosshair cursor over the clickable road.
function injectStyle() {
  if (document.getElementById('gc-drive-style')) return;
  const style = document.createElement('style');
  style.id = 'gc-drive-style';
  style.textContent = '.gc-drive canvas { cursor: crosshair; touch-action: manipulation; }';
  document.head.appendChild(style);
}

const HAZARDS = [
  { key: 'cone', label: 'Cone' },
  { key: 'pedestrian', label: 'Pedestrian' },
  { key: 'car', label: 'Stalled car' },
];

export async function mount(el, { decide, mode } = {}) {
  injectStyle();
  const refs = mountChrome(el, { label: 'Drive · 3-lane road' });
  el.classList.add('gc-drive');
  const canvas = refs.canvas;
  const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

  // data/replays/drive.json is {frames, summary} (scripts/record_games.mjs) - replayFrame()
  // wants the bare frames array.
  const replay = ((await loadJSON('data/replays/drive.json').catch(() => null)) ?? {}).frames ?? [];

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));

  const { scene, road } = buildScene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 300);

  const { group: egoMesh, brakeMat } = buildEgo();
  scene.add(egoMesh);

  const shadowTex = buildShadowTexture();
  const egoShadow = buildShadow(shadowTex, 2.2, 4.4);
  scene.add(egoShadow);

  let engine = new Drive({ seed: Math.floor(Math.random() * 1e9) });
  const lightMeshes = engine.lights.map((l) => buildLight(l.pos));
  lightMeshes.forEach(({ group }) => scene.add(group));

  const trafficPool = [];
  const trafficShadowPool = [];
  const pedPool = [];
  const conePool = [];
  function ensurePool(pool, builder, count) {
    while (pool.length < count) {
      const mesh = builder();
      scene.add(mesh);
      pool.push(mesh);
    }
    for (let i = 0; i < pool.length; i++) pool[i].visible = i < count;
    return pool;
  }

  let policyName = 'model';
  let prevState = engine.state();
  let currState = engine.state();
  let lastTickAt = performance.now();
  let replayIndex = 0;
  let lastDecision = { candidates: engine.candidates(), probs: {}, p_null: null, sentence: engine.describe() };
  const human = createHumanOverride(3000);
  let pendingRestart = false;

  function restart() {
    pendingRestart = false;
    engine = new Drive({ seed: Math.floor(Math.random() * 1e9) });
    // light post positions are fixed by construction (same LIGHT_SPACING every reset) so the
    // existing meshes stay valid - only their state/timer (and therefore lamp colour) changes.
    prevState = engine.state();
    currState = engine.state();
    replayIndex = 0;
    lastTickAt = performance.now();
    lastDecision = { candidates: engine.candidates(), probs: {}, p_null: null, sentence: engine.describe() };
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
      if (currState.done) {
        pendingRestart = true;
        setTimeout(restart, 900);
      }
      return;
    }

    if (engine.done) return;
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
      move = legal.includes(human.move) ? human.move : legal.includes('hold speed') ? 'hold speed' : legal[0];
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
    if (engine.done) {
      pendingRestart = true;
      setTimeout(restart, 900);
    }
  }

  function resize() {
    const rect = refs.stage.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    renderer.setSize(w, h, false);
    const aspect = w / h;
    // viewSize picked so the 9.6m-wide road fills ~60% of the card width at the card's live
    // aspect (~1.18): 9.6 / (0.6 * 1.18) ~= 13.6. Paired with CAM_HEIGHT/CAM_BACK/CAM_AHEAD
    // below (a shallower camera than a pure top-down one) so ~55m ahead/~15m behind still fit
    // in that same vertical frustum - full ortho math ties width and depth to one viewSize, so
    // 60% width and 120m/30m depth (the original ask) don't both fit without the camera going
    // near-horizontal, which makes tall props (light posts, cars) float off their true depth.
    const viewSize = 15;
    camera.left = (-viewSize * aspect) / 2;
    camera.right = (viewSize * aspect) / 2;
    camera.top = viewSize / 2;
    camera.bottom = -viewSize / 2;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(refs.stage);
  resize();

  function draw() {
    // prefers-reduced-motion: skip the sub-tick interpolation (camera/road jump straight to the
    // current tick instead of easing) rather than turning off ticking itself.
    const t = reducedMotion ? 1 : Math.max(0, Math.min(1, (performance.now() - lastTickAt) / TICK_MS));
    const pe = prevState.ego;
    const ce = currState.ego;
    const egoX = lerp(laneToX(pe.lane), laneToX(ce.lane), t);
    const egoZ = lerp(pe.position, ce.position, t);
    egoMesh.position.set(egoX, 0, egoZ);
    egoShadow.position.set(egoX, 0.006, egoZ);
    brakeMat.color.set(ce.speed < pe.speed ? TOKENS.paper : TOKENS.ink);

    camera.position.set(egoX, CAM_HEIGHT, egoZ - CAM_BACK);
    camera.lookAt(egoX, 0, egoZ + CAM_AHEAD);

    const traffic = ensurePool(trafficPool, buildTrafficCar, currState.traffic.length);
    const trafficShadows = ensurePool(trafficShadowPool, () => buildShadow(shadowTex, 2, 4), currState.traffic.length);
    currState.traffic.forEach((car, i) => {
      const prev = prevState.traffic[i] ?? car;
      const x = lerp(laneToX(prev.lane), laneToX(car.lane), t);
      const z = lerp(prev.position, car.position, t);
      traffic[i].position.set(x, 0.55, z);
      trafficShadows[i].position.set(x, 0.006, z);
    });

    // ponytail: pedestrians and placed cones spawn/despawn irregularly (no stable index to
    // interpolate against) - snap to the current frame's positions rather than tracking identity
    // across ticks, same call already made for pedestrians.
    const peds = ensurePool(pedPool, buildPedestrian, currState.pedestrians.length);
    currState.pedestrians.forEach((p, i) => {
      peds[i].position.set(laneToX(p.lane), 0.7, p.pos);
    });

    const cones = ensurePool(conePool, buildCone, currState.cones.length);
    currState.cones.forEach((c, i) => {
      cones[i].position.set(laneToX(c.lane), 0.4, c.pos);
    });

    lightMeshes.forEach(({ lamp }, i) => {
      const l = currState.lights[i];
      if (!l) return;
      lamp.material.color.set(l.state === 'green' ? TOKENS.paper : TOKENS.ink);
    });

    renderer.render(scene, camera);
  }

  function frame() {
    draw();
    refs.hud.innerHTML = '';
    const l1 = document.createElement('div');
    l1.textContent = 'Drive · 3-lane road';
    const l2 = document.createElement('div');
    l2.textContent = `${currState.ego.speed} km/h · lane ${currState.ego.lane} · exit in ${Math.max(0, Math.round(DEST_POS - currState.ego.position))} m`;
    refs.hud.append(l1, l2);
    paintDecision(refs, lastDecision);
    rafId = requestAnimationFrame(frame);
  }
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

  // (a) Interactivity: click/tap the road to place a hazard. The clicked point is raycast onto
  // the `road` mesh (its extent is exactly lanes 1..LANES, so a hit off the road never happens)
  // and converted to a lane + world position, then handed to Drive.placeHazard() - the engine
  // decides legality (on-road, ahead of the car) and, once accepted, the hazard flows through
  // the same describe() sentence used for procedurally spawned pedestrians/traffic, so the next
  // tick's model call sees one of the pre-computed sentences and the decision bars move.
  let hazardKind = HAZARDS[0].key;
  const hazardHint = document.createElement('div');
  hazardHint.className = 'gc-sentence';
  refs.sentence.parentElement.appendChild(hazardHint);
  function updateHazardHint() {
    const label = HAZARDS.find((h) => h.key === hazardKind).label.toLowerCase();
    hazardHint.textContent = `Click the road to place a ${label}.`;
  }
  updateHazardHint();

  const hazardControls = el.querySelector('.gc-controls');
  const hazardBtns = HAZARDS.map(({ key, label }) => {
    const b = document.createElement('button');
    b.className = 'btn outline gc-btn';
    b.type = 'button';
    b.textContent = label;
    b.addEventListener('click', () => {
      hazardKind = key;
      hazardBtns.forEach((btn, i) => btn.classList.toggle('gc-on', HAZARDS[i].key === hazardKind));
      updateHazardHint();
    });
    hazardControls?.insertBefore(b, hazardControls.firstChild);
    return b;
  });
  hazardBtns[0]?.classList.add('gc-on');

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  function placeAt(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(road)[0];
    if (!hit) return; // off-road click - ignored, per spec
    const lane = Math.round(hit.point.x / LANE_W + (LANES + 1) / 2);
    engine.placeHazard(hazardKind, lane, Math.round(hit.point.z)); // no-ops (returns false) behind the car
  }
  canvas.addEventListener('click', (e) => placeAt(e.clientX, e.clientY));

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
