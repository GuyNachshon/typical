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

function buildScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(TOKENS.putty);

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

  const road = new THREE.Mesh(
    new THREE.PlaneGeometry(LANE_W * LANES, ROAD_LENGTH + 100),
    new THREE.MeshLambertMaterial({ color: TOKENS.graphite })
  );
  road.rotation.x = -Math.PI / 2;
  road.position.set(0, 0.01, ROAD_LENGTH / 2);
  scene.add(road);

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

  return { scene, sun };
}

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
  return group;
}

function buildTrafficCar() {
  return new THREE.Mesh(
    new THREE.BoxGeometry(1.7, 1.1, 3.4),
    new THREE.MeshLambertMaterial({ color: TOKENS.bone, flatShading: true })
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

export async function mount(el, { decide, mode } = {}) {
  const refs = mountChrome(el, { label: 'Drive · 3-lane road' });
  const canvas = refs.canvas;

  // data/replays/drive.json is {frames, summary} (scripts/record_games.mjs) - replayFrame()
  // wants the bare frames array.
  const replay = ((await loadJSON('data/replays/drive.json').catch(() => null)) ?? {}).frames ?? [];

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));

  const { scene } = buildScene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 300);

  const egoMesh = buildEgo();
  scene.add(egoMesh);

  let engine = new Drive({ seed: Math.floor(Math.random() * 1e9) });
  const lightMeshes = engine.lights.map((l) => buildLight(l.pos));
  lightMeshes.forEach(({ group }) => scene.add(group));

  const trafficPool = [];
  const pedPool = [];
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
    const t = Math.max(0, Math.min(1, (performance.now() - lastTickAt) / TICK_MS));
    const pe = prevState.ego;
    const ce = currState.ego;
    const egoX = lerp(laneToX(pe.lane), laneToX(ce.lane), t);
    const egoZ = lerp(pe.position, ce.position, t);
    egoMesh.position.set(egoX, 0, egoZ);

    camera.position.set(egoX, CAM_HEIGHT, egoZ - CAM_BACK);
    camera.lookAt(egoX, 0, egoZ + CAM_AHEAD);

    const traffic = ensurePool(trafficPool, buildTrafficCar, currState.traffic.length);
    currState.traffic.forEach((car, i) => {
      const prev = prevState.traffic[i] ?? car;
      const x = lerp(laneToX(prev.lane), laneToX(car.lane), t);
      const z = lerp(prev.position, car.position, t);
      traffic[i].position.set(x, 0.55, z);
    });

    // ponytail: pedestrians spawn/despawn irregularly (no stable index to interpolate against)
    // - snap to the current frame's positions rather than tracking identity across ticks.
    const peds = ensurePool(pedPool, buildPedestrian, currState.pedestrians.length);
    currState.pedestrians.forEach((p, i) => {
      peds[i].position.set(laneToX(p.lane), 0.7, p.pos);
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
