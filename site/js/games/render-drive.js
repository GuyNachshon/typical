// Three.js chase-cam renderer over js/games/drive.js's pure engine: a JevPilot-style daylight
// city street (asphalt, kerbs, sidewalks, buildings, street lamps), a sedan ego with wheels and
// brake lights, and an "intent ribbon" that draws the model's chosen manoeuvre on the road ahead
// of the car with its winning probability printed at the tip.
//
// Daylight exception (client-directed): TOKENS is the site's dark-scene palette and stays
// exactly as-is for the chrome (HUD/bars/buttons, all styled from exhibits.css - untouched by
// this file). The 3D scene below deliberately does NOT use TOKENS: a bright, real-world street
// palette is what was asked for after the dark version read as an empty void. Do not "fix" this
// back to TOKENS; it is intentional.
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js';
import { Drive, greedyPolicy, QUESTION, ROAD_LENGTH, DEST_POS } from './drive.js';
import { mountChrome, paintDecision, watchVisibility, createTicker, createHumanOverride, bindKeys, modelPolicy, replayFrame, loadJSON } from './loop.js';

const TICK_MS = 400;
// Mirrors js/games/drive.js's internal constants (not exported - small fixed numbers, not
// worth threading through a second module for).
const LANES = 3;
const LANE_W = 3.2;

// Chase camera: low and close, pulled back just far enough to clear the sedan's own body, tuned
// (by projecting the ego's world position through this exact camera, not by eye) so the car
// lands ~60% down the frame with the street receding into the upper third. See the composition
// note in mount()'s resize() for the viewSize math.
const CAM_HEIGHT = 4.6; // metres above the road
const CAM_BACK = 15.5; // metres behind the ego
const CAM_AHEAD = 34; // metres ahead the camera aims at
const CAM_OFFSET_X = 3.4; // metres the camera rides left of the ego
const CAM_LOOK_Y = 0.7; // aim low: a camera tilted down puts the car higher in frame, clear of the readout
const SKY = '#bcd6e6';
const ASPHALT = '#6b6b6e';
const KERB = '#b8b2a6';
const SIDEWALK = '#cfc9bd';
const LANE_MARK = '#f2f2ee';
const POST_DARK = '#2b2b2e';
const LIGHT_RED = '#e2453b';
const LIGHT_GREEN = '#3fae5c';
const LAMP_GLOW = '#ffe9a8';
const CONE_ORANGE = '#e2691b';
const PED_CLOTHES = '#3d5a80';
const PED_SKIN = '#e0ac69';
const WHEEL_DARK = '#1c1c1e';
const CABIN_DARK = '#33383d';
const GLASS_TINT = '#aeb9c2';
const EGO_BODY = '#f2f1ec';
const TAIL_OFF = '#7a2b28';
const TAIL_ON = '#ff3b30';
const HAZARD_WARN = '#e2691b';
const INTENT_COLOR = '#ffb100';
const TRAFFIC_COLORS = ['#b23b3b', '#2e5fa3', '#3f8f5f', '#c9c9c9', '#8a7fae'];
const FOG_NEAR = 14;
const FOG_FAR = 85;

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

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// Soft dark disc (contact shadow / cheap AO) reused across every ego/traffic instance.
function buildShadowTexture() {
  const size = 64;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, 'rgba(0,0,0,0.4)');
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

// A thin flat ring on the ground - the only marker for a hazard the *visitor* placed (never a
// procedurally spawned one), so the visitor can see exactly what they just did.
function buildRing() {
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.6, 0.85, 28),
    new THREE.MeshBasicMaterial({ color: '#ffffff', side: THREE.DoubleSide, transparent: true, opacity: 0.95, depthTest: false })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.renderOrder = 1;
  return ring;
}

// Two stacked boxes (body + cabin) is the low-poly "sedan" the brief asks for: rounded enough to
// read as a car, four wheels, a tinted windscreen, and rear lamps that light on braking (ego
// only - `tailMat` is the material both lamps share, toggled in draw()).
function buildCar(bodyColor, cabinColor) {
  const group = new THREE.Group();
  const lower = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.55, 4.2), new THREE.MeshLambertMaterial({ color: bodyColor, flatShading: true }));
  lower.position.y = 0.275;
  group.add(lower);

  const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.55, 0.5, 2.5), new THREE.MeshLambertMaterial({ color: cabinColor, flatShading: true }));
  cabin.position.set(0, 0.8, -0.2);
  group.add(cabin);

  const glass = new THREE.Mesh(new THREE.PlaneGeometry(1.3, 0.4), new THREE.MeshBasicMaterial({ color: GLASS_TINT }));
  glass.position.set(0, 0.86, 1.02);
  glass.rotation.x = -Math.PI / 2.6;
  group.add(glass);

  const wheelMat = new THREE.MeshLambertMaterial({ color: WHEEL_DARK });
  const wheelGeo = new THREE.CylinderGeometry(0.32, 0.32, 0.26, 10);
  for (const x of [-0.88, 0.88]) {
    for (const z of [-1.55, 1.55]) {
      const wheel = new THREE.Mesh(wheelGeo, wheelMat);
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(x, 0.32, z);
      group.add(wheel);
    }
  }

  const tailMat = new THREE.MeshBasicMaterial({ color: TAIL_OFF });
  const tailGeo = new THREE.BoxGeometry(0.34, 0.16, 0.06);
  for (const x of [-0.62, 0.62]) {
    const lamp = new THREE.Mesh(tailGeo, tailMat);
    lamp.position.set(x, 0.5, -2.12);
    group.add(lamp);
  }
  return { group, tailMat };
}

function buildCone() {
  const group = new THREE.Group();
  const body = new THREE.Mesh(new THREE.ConeGeometry(0.28, 0.7, 8), new THREE.MeshLambertMaterial({ color: CONE_ORANGE, flatShading: true }));
  body.position.y = 0.35;
  group.add(body);
  const band = new THREE.Mesh(
    new THREE.CylinderGeometry(0.19, 0.22, 0.12, 8, 1, true),
    new THREE.MeshLambertMaterial({ color: '#ffffff', side: THREE.DoubleSide })
  );
  band.position.y = 0.4;
  group.add(band);
  return group;
}

function buildPedestrian() {
  const group = new THREE.Group();
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.24, 0.55, 4, 8), new THREE.MeshLambertMaterial({ color: PED_CLOTHES }));
  body.position.y = 0.55;
  group.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.16, 8, 6), new THREE.MeshLambertMaterial({ color: PED_SKIN }));
  head.position.y = 1.0;
  group.add(head);
  return group;
}

function buildLight(pos) {
  const group = new THREE.Group();
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 3.2, 8), new THREE.MeshLambertMaterial({ color: POST_DARK }));
  post.position.y = 1.6;
  const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.3, 0.3), new THREE.MeshBasicMaterial({ color: LIGHT_RED }));
  lamp.position.y = 3.3;
  group.add(post, lamp);
  const x = laneToX(LANES) + LANE_W / 2 + 0.6;
  group.position.set(x, 0, pos);
  return { group, lamp };
}

// Simple extruded block with a two-tone "glass-grid" facade (a handful of contrasting horizontal
// bands, not per-window geometry) - flat rectangles in two tones, no textures, per the brief.
function buildBuilding(w, d, h, baseColor, glassColor) {
  const group = new THREE.Group();
  const base = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshLambertMaterial({ color: baseColor, flatShading: true }));
  base.position.y = h / 2;
  group.add(base);
  const bands = Math.max(2, Math.round(h / 4));
  const glassMat = new THREE.MeshLambertMaterial({ color: glassColor });
  for (let i = 0; i < bands; i++) {
    const band = new THREE.Mesh(new THREE.BoxGeometry(w * 0.94, h / (bands * 2.2), d * 1.01), glassMat);
    band.position.y = ((i + 0.5) / bands) * h;
    group.add(band);
  }
  return group;
}

// Black post + curved arm (approximated as a short horizontal box) + a glowing head - these are
// what give speed away as they stream past at regular intervals.
function buildLampPost(side) {
  const group = new THREE.Group();
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 4.2, 6), new THREE.MeshLambertMaterial({ color: POST_DARK }));
  post.position.y = 2.1;
  group.add(post);
  const arm = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.06, 0.06), new THREE.MeshLambertMaterial({ color: POST_DARK }));
  arm.position.set(-side * 0.45, 4.0, 0);
  group.add(arm);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 6), new THREE.MeshBasicMaterial({ color: LAMP_GLOW }));
  head.position.set(-side * 0.9, 3.9, 0);
  group.add(head);
  return group;
}

// One small sidewalk prop, alternating kind - breaks the lamp-post repetition at long intervals.
function buildStreetProp(kind) {
  const group = new THREE.Group();
  if (kind === 'bench') {
    const seat = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.08, 0.4), new THREE.MeshLambertMaterial({ color: '#6b5a4a' }));
    seat.position.y = 0.42;
    group.add(seat);
    const legMat = new THREE.MeshLambertMaterial({ color: POST_DARK });
    for (const x of [-0.5, 0.5]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.42, 0.35), legMat);
      leg.position.set(x, 0.21, 0);
      group.add(leg);
    }
  } else {
    const bin = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.18, 0.55, 8), new THREE.MeshLambertMaterial({ color: POST_DARK }));
    bin.position.y = 0.28;
    group.add(bin);
  }
  return group;
}

// Deterministic small LCG so the street furniture is varied but stable across reloads (decor
// only - never touches the engine's own seeded RNG).
function makeRand(seed) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return (s % 1000) / 1000;
  };
}

function buildScene() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(SKY);
  scene.fog = new THREE.Fog(SKY, FOG_NEAR, FOG_FAR);

  scene.add(new THREE.AmbientLight('#ffffff', 0.95));
  const sun = new THREE.DirectionalLight('#fff6e2', 0.55);
  sun.position.set(-20, 40, -10);
  scene.add(sun);

  const roadHalfW = (LANE_W * LANES) / 2;
  const kerbW = 0.22;
  const sidewalkW = 2.4;
  const far = ROAD_LENGTH + 100;
  const rand = makeRand(7);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(90, far), new THREE.MeshLambertMaterial({ color: SIDEWALK }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(0, -0.02, ROAD_LENGTH / 2);
  scene.add(ground);

  const road = new THREE.Mesh(new THREE.PlaneGeometry(roadHalfW * 2, far), new THREE.MeshLambertMaterial({ color: ASPHALT }));
  road.rotation.x = -Math.PI / 2;
  road.position.set(0, 0.01, ROAD_LENGTH / 2);
  scene.add(road);

  const kerbMat = new THREE.MeshLambertMaterial({ color: KERB });
  const sidewalkMat = new THREE.MeshLambertMaterial({ color: SIDEWALK });
  for (const side of [-1, 1]) {
    const kerb = new THREE.Mesh(new THREE.BoxGeometry(kerbW, 0.12, far), kerbMat);
    kerb.position.set(side * (roadHalfW + kerbW / 2), 0.05, ROAD_LENGTH / 2);
    scene.add(kerb);
    const walk = new THREE.Mesh(new THREE.PlaneGeometry(sidewalkW, far), sidewalkMat);
    walk.rotation.x = -Math.PI / 2;
    walk.position.set(side * (roadHalfW + kerbW + sidewalkW / 2), 0.015, ROAD_LENGTH / 2);
    scene.add(walk);
  }

  // solid white edge lines + dashed white internal lane boundaries
  const markMat = new THREE.MeshBasicMaterial({ color: LANE_MARK });
  for (const x of [-roadHalfW + 0.08, roadHalfW - 0.08]) {
    const line = new THREE.Mesh(new THREE.PlaneGeometry(0.14, far), markMat);
    line.rotation.x = -Math.PI / 2;
    line.position.set(x, 0.02, ROAD_LENGTH / 2);
    scene.add(line);
  }
  const dashGeo = new THREE.BoxGeometry(0.16, 0.03, 1.6);
  const dashMat = new THREE.MeshBasicMaterial({ color: LANE_MARK });
  const boundaries = [laneToX(1) + LANE_W / 2, laneToX(2) + LANE_W / 2];
  const dashSpacing = 6;
  const dashCount = boundaries.length * Math.ceil(ROAD_LENGTH / dashSpacing);
  const dashes = new THREE.InstancedMesh(dashGeo, dashMat, dashCount);
  let di = 0;
  const m4 = new THREE.Matrix4();
  for (const x of boundaries) {
    for (let z = 0; z < ROAD_LENGTH; z += dashSpacing) {
      m4.makeTranslation(x, 0.03, z);
      dashes.setMatrixAt(di++, m4);
    }
  }
  dashes.instanceMatrix.needsUpdate = true;
  scene.add(dashes);

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
  scene.add(new THREE.Mesh(rampGeo, new THREE.MeshLambertMaterial({ color: ASPHALT, side: THREE.DoubleSide })));

  // buildings, street lamps and the occasional prop along both sidewalks
  const buildingX = roadHalfW + kerbW + sidewalkW + 2;
  const lampX = roadHalfW + kerbW + sidewalkW - 0.4;
  const palette = [
    ['#8b93a0', '#5c6672'], ['#9a9186', '#6b6259'], ['#7f8b86', '#526059'], ['#93888b', '#645a5d'],
  ];
  for (let z = 20; z < ROAD_LENGTH + 60; z += 42) {
    for (const side of [-1, 1]) {
      const [base, glass] = palette[Math.floor(rand() * palette.length)];
      const h = 8 + rand() * 22;
      const w = 8 + rand() * 6;
      const d = 8 + rand() * 6;
      const b = buildBuilding(w, d, h, base, glass);
      b.position.set(side * (buildingX + w / 2 + rand() * 4), 0, z + rand() * 10);
      scene.add(b);
    }
  }
  for (let z = 10; z < ROAD_LENGTH + 40; z += 26) {
    for (const side of [-1, 1]) {
      const lamp = buildLampPost(side);
      lamp.position.set(side * lampX, 0, z);
      scene.add(lamp);
    }
  }
  for (let z = 30, side = -1; z < ROAD_LENGTH; z += 130, side *= -1) {
    const prop = buildStreetProp(rand() > 0.5 ? 'bench' : 'bin');
    prop.position.set(side * (lampX - 0.9), 0, z);
    scene.add(prop);
  }

  return { scene, road };
}

// Label sprite for the intent ribbon's tip - a small rounded pill with the winning action's
// probability (or the action name for a forced 1-candidate tick).
function buildLabelSprite() {
  const c = document.createElement('canvas');
  c.width = 160;
  c.height = 72;
  const ctx = c.getContext('2d');
  const tex = new THREE.CanvasTexture(c);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
  sprite.scale.set(1.3, 0.6, 1);
  let last = null;
  function update(text) {
    if (text === last) return;
    last = text;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = 'rgba(24,20,12,0.82)';
    roundRect(ctx, 4, 14, c.width - 8, c.height - 28, 12);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 34px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, c.width / 2, c.height / 2 + 1);
    tex.needsUpdate = true;
  }
  return { sprite, update };
}

// Local-space (relative to the ego) path for the chosen action: a straight ribbon ahead for
// accelerate/hold speed, a shorter one for follow, a short blunt stub for brake/stop, and a
// curve into the target lane for a lane change.
function ribbonPoints(action) {
  const front = 2.3;
  if (action === 'change lane left' || action === 'change lane right') {
    const dir = action === 'change lane left' ? -1 : 1;
    return [
      new THREE.Vector3(0, 0.04, front),
      new THREE.Vector3((dir * LANE_W) / 2, 0.04, front + 6),
      new THREE.Vector3(dir * LANE_W, 0.04, front + 12),
    ];
  }
  if (action === 'brake' || action === 'stop') return [new THREE.Vector3(0, 0.04, front), new THREE.Vector3(0, 0.04, front + 3)];
  if (action === 'follow') return [new THREE.Vector3(0, 0.04, front), new THREE.Vector3(0, 0.04, front + 8)];
  return [new THREE.Vector3(0, 0.04, front), new THREE.Vector3(0, 0.04, front + 14)]; // accelerate / hold speed
}

function buildRibbon(action) {
  const pts = ribbonPoints(action);
  const curve = new THREE.CatmullRomCurve3(pts);
  const blunt = action === 'brake' || action === 'stop';
  const tube = new THREE.TubeGeometry(curve, blunt ? 4 : 16, blunt ? 0.3 : 0.2, 8, false);
  const mesh = new THREE.Mesh(tube, new THREE.MeshBasicMaterial({ color: INTENT_COLOR }));
  const tip = pts[pts.length - 1];
  // The label always sits at least 6 m ahead of the ribbon's start, regardless of how short a
  // blunt brake/stop stub is - otherwise, at this zoomed-in a camera, a short ribbon's tip (and
  // the always-on-top label sprite riding it) lands close enough to read as sitting on the car.
  const labelPos = tip.clone();
  labelPos.z = Math.max(tip.z, pts[0].z + 6);
  return { mesh, tip: labelPos };
}

// Injected once (scoped to .gc-drive, this module's own root class - never leaks into the
// snake/doom cards which share loop.js's .gc-root). Two things no existing exhibits.css class
// covers: a crosshair cursor over the clickable road, and a home for the hazard-chip row - it
// docks above loop.js's Model/Restart row (same right edge, same small-button language) rather
// than joining .gc-controls's own no-wrap flex line, labelled and wrapping/shrinking on narrow
// panels so it never pushes past the panel edge or collides with Model/Restart.
function injectStyle() {
  if (document.getElementById('gc-drive-style')) return;
  const style = document.createElement('style');
  style.id = 'gc-drive-style';
  style.textContent = `
    .gc-drive canvas { cursor: crosshair; touch-action: manipulation; }
    .gc-drive .gc-hazards-wrap {
      position: absolute; right: 18px; bottom: 54px; z-index: 4;
      display: flex; flex-direction: column; align-items: flex-end; gap: 4px;
      max-width: min(240px, calc(100% - 36px));
    }
    .gc-drive .gc-hazards-label {
      font-family: var(--font-mono, monospace); font-size: 10px; letter-spacing: 0.36px;
      text-transform: uppercase; color: rgba(255,255,255,0.85); text-shadow: 0 1px 3px rgba(0,0,0,0.7);
    }
    .gc-drive .gc-hazards { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px; }
  `;
  document.head.appendChild(style);
}

const HAZARDS = [
  { key: 'cone', label: 'Cone', name: 'cone' },
  { key: 'pedestrian', label: 'Ped', name: 'pedestrian' },
  { key: 'car', label: 'Car', name: 'stopped car' },
];

export async function mount(el, { decide, mode } = {}) {
  injectStyle();
  const refs = mountChrome(el, { label: 'Drive · city street' });
  el.classList.add('gc-drive');
  const canvas = refs.canvas;
  const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

  // data/replays/drive.json is {frames, summary} (scripts/record_games.mjs) - replayFrame()
  // wants the bare frames array.
  const replay = ((await loadJSON('data/replays/drive.json').catch(() => null)) ?? {}).frames ?? [];

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));

  const { scene, road } = buildScene();
  // Perspective, not orthographic. An ortho chase camera gives a road with no convergence: the
  // lanes stay parallel to the frame edge and the scene reads as a flat collage of blocks, which
  // is exactly how it looked. A vanishing point is most of what makes a driving shot legible.
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 400);

  const { group: egoMesh, tailMat } = buildCar(EGO_BODY, CABIN_DARK);
  scene.add(egoMesh);

  const shadowTex = buildShadowTexture();
  const egoShadow = buildShadow(shadowTex, 2.1, 4.6);
  scene.add(egoShadow);

  const ribbonGroup = new THREE.Group();
  scene.add(ribbonGroup);
  const label = buildLabelSprite();
  ribbonGroup.add(label.sprite);
  let ribbonMesh = null;
  let ribbonKey = null;

  let engine = new Drive({ seed: Math.floor(Math.random() * 1e9) });
  const lightMeshes = engine.lights.map((l) => buildLight(l.pos));
  lightMeshes.forEach(({ group }) => scene.add(group));

  const trafficPool = [];
  const trafficShadowPool = [];
  const trafficRingPool = [];
  const pedPool = [];
  const pedRingPool = [];
  const conePool = [];
  const coneRingPool = [];
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
    // viewSize picked (by projecting the ego through CAM_HEIGHT/CAM_BACK/CAM_AHEAD above, not by
    // eye) so the sedan lands ~55-65% down the frame at a readable size on both the desktop card
    // (~620x560, aspect ~1.1) and the mobile one (~350x360, aspect ~0.97), while the ~1.3 lanes
    // either side of the ego needed for a legible lane change still fit at the tighter mobile
    // aspect. A larger viewSize (more zoomed out) shrinks the car below "readable"; a smaller one
    // clips the neighbour lane on mobile.
    camera.aspect = aspect;
    // Narrow cards (the mobile panel is nearly square) crop horizontally, so widen the vertical
    // field there to keep both neighbour lanes in shot.
    camera.fov = aspect < 1.15 ? 52 : 42;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(refs.stage);
  resize();

  let settle = 0; // eased camera dip under braking/acceleration - lerped every frame, never snaps (no shake)
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
    tailMat.color.set(ce.speed < pe.speed ? TAIL_ON : TAIL_OFF);

    const targetSettle = reducedMotion ? 0 : (pe.speed - ce.speed) * 0.015; // >0 while decelerating
    settle = lerp(settle, targetSettle, 0.12);
    // The camera sits behind and above the ego and looks down the road. The car lands high enough
    // in the frame to clear the decision readout (.gc-foot, bottom-left) without panning the whole
    // scene sideways — panning was what skewed the shot.
    // A modest lateral offset (the camera rides to the left of the car) puts the ego in the right
    // half of the frame, clear of the decision bars that dock bottom-left. With a perspective
    // camera this reads as an offset chase view; the earlier ortho pan just sheared the scene.
    const camX = egoX * 0.6 - CAM_OFFSET_X;
    camera.position.set(camX, CAM_HEIGHT + settle * 0.4, egoZ - CAM_BACK - settle * 1.2);
    camera.lookAt(camX + CAM_OFFSET_X * 0.45, CAM_LOOK_Y, egoZ + CAM_AHEAD);

    // intent ribbon: the model's own chosen action, drawn on the road ahead of the car, with its
    // winning probability at the tip - rebuilt only when the action changes, not every frame.
    ribbonGroup.position.set(egoX, 0, egoZ);
    const probs = lastDecision.probs ?? {};
    const ranked = Object.entries(probs).sort((a, b) => b[1] - a[1])[0];
    const action = ranked ? ranked[0] : lastDecision.candidates?.[0] ?? 'hold speed';
    const prob = ranked ? ranked[1] : lastDecision.candidates?.length === 1 ? 1 : null;
    if (action !== ribbonKey) {
      if (ribbonMesh) {
        ribbonGroup.remove(ribbonMesh.mesh);
        ribbonMesh.mesh.geometry.dispose();
      }
      ribbonMesh = buildRibbon(action);
      ribbonGroup.add(ribbonMesh.mesh);
      label.sprite.position.copy(ribbonMesh.tip).add(new THREE.Vector3(0, 1.15, 0));
      ribbonKey = action;
    }
    label.update(prob != null ? `${Math.round(prob * 100)}%` : action);

    const traffic = ensurePool(trafficPool, () => {
      const placed = currState.traffic[trafficPool.length]?.placed;
      const color = placed ? HAZARD_WARN : TRAFFIC_COLORS[trafficPool.length % TRAFFIC_COLORS.length];
      return buildCar(color, CABIN_DARK).group; // traffic tail lights stay off (only the ego's brake to the model's decision)
    }, currState.traffic.length);
    const trafficShadows = ensurePool(trafficShadowPool, () => buildShadow(shadowTex, 2, 4.2), currState.traffic.length);
    const trafficRings = ensurePool(trafficRingPool, buildRing, currState.traffic.length);
    currState.traffic.forEach((car, i) => {
      const prev = prevState.traffic[i] ?? car;
      const x = lerp(laneToX(prev.lane), laneToX(car.lane), t);
      const z = lerp(prev.position, car.position, t);
      traffic[i].position.set(x, 0, z);
      trafficShadows[i].position.set(x, 0.006, z);
      trafficRings[i].position.set(x, 0.012, z);
      trafficRings[i].visible = Boolean(car.placed);
    });

    // ponytail: pedestrians and placed cones spawn/despawn irregularly (no stable index to
    // interpolate against) - snap to the current frame's positions rather than tracking identity
    // across ticks.
    const peds = ensurePool(pedPool, buildPedestrian, currState.pedestrians.length);
    const pedRings = ensurePool(pedRingPool, buildRing, currState.pedestrians.length);
    currState.pedestrians.forEach((p, i) => {
      peds[i].position.set(laneToX(p.lane), 0, p.pos);
      pedRings[i].position.set(laneToX(p.lane), 0.012, p.pos);
      pedRings[i].visible = Boolean(p.placed);
    });

    const cones = ensurePool(conePool, buildCone, currState.cones.length);
    const coneRings = ensurePool(coneRingPool, buildRing, currState.cones.length);
    currState.cones.forEach((c, i) => {
      cones[i].position.set(laneToX(c.lane), 0, c.pos);
      coneRings[i].position.set(laneToX(c.lane), 0.012, c.pos);
      coneRings[i].visible = Boolean(c.placed);
    });

    lightMeshes.forEach(({ lamp }, i) => {
      const l = currState.lights[i];
      if (!l) return;
      lamp.material.color.set(l.state === 'green' ? LIGHT_GREEN : LIGHT_RED);
    });

    renderer.render(scene, camera);
  }

  function frame() {
    draw();
    refs.hud.innerHTML = '';
    const l1 = document.createElement('div');
    l1.textContent = 'Drive · city street';
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
  // tick's model call sees one of the pre-computed sentences and the decision bars move. A
  // marker ring (drawn above) appears under anything the visitor placed.
  // The hint lives in the hazard-chip label (bottom-right), not in `.gc-foot` (bottom-left) -
  // .gc-foot's height is part of the "keep clear of the readout" budget the composition is tuned
  // against, so nothing of mine grows it.
  let hazardKind = HAZARDS[0].key;
  const hazardWrap = document.createElement('div');
  hazardWrap.className = 'gc-hazards-wrap';
  const hazardLabel = document.createElement('div');
  hazardLabel.className = 'gc-hazards-label';
  function updateHazardHint() {
    const name = HAZARDS.find((h) => h.key === hazardKind).name;
    hazardLabel.textContent = `Click the road to place a ${name}`;
  }
  updateHazardHint();
  const hazardRow = document.createElement('div');
  hazardRow.className = 'gc-hazards';
  const hazardBtns = HAZARDS.map(({ key, label: btnLabel }) => {
    const b = document.createElement('button');
    b.className = 'btn outline gc-btn';
    b.type = 'button';
    b.textContent = btnLabel;
    b.addEventListener('click', () => {
      hazardKind = key;
      hazardBtns.forEach((btn, i) => btn.classList.toggle('gc-on', HAZARDS[i].key === hazardKind));
      updateHazardHint();
    });
    hazardRow.appendChild(b);
    return b;
  });
  hazardBtns[0].classList.add('gc-on');
  hazardWrap.append(hazardLabel, hazardRow);
  el.appendChild(hazardWrap);

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
