// filmfx.js — the hero film shown on a CRT.
//
// The frame is not filtered in place: it is redrawn here, strip by strip, with a barrel warp, and
// the iframe behind is left running but invisible. That matters for two reasons. A canvas can be
// warped and an iframe cannot, and compositing anything on top of a live iframe put two different
// frames on screen at once — the WebGL buffer we read back can lag the one being displayed, which
// is what produced the ghost of a previous camera angle.
//
// Drawn per frame, into one canvas:
//   1. the frame, in horizontal strips, each swelled by its distance from the centre
//   2. bloom: the same frame blurred through a curve so only the lit parts survive, added back
//   3. scanlines, and a much fainter vertical grille behind them
//   4. vignette, then the tube mask — rounded corners and the dark surround outside the glass
//   5. a little grain
//
// A decision that rewrites the probability panel calls pulse(): ~420ms of extra contrast, bloom
// and scanline depth that settles back on its own.
//
//   const fx = mountFilmFx(host, getSource, { after });  fx.pulse();  fx.stop();

const FPS = 26; // the game ticks at ~2.5 decisions/s; this is a film rate, not a game rate
const STRIPS = 64; // horizontal slices the warp is built from
const BULGE = 0.05; // how far the glass swells at the centre, as a share of width
const PULSE_MS = 320; // decisions land every ~380ms, so a longer pulse would never finish and the
// film would read as permanently graded instead of pulsing once per decision

function reduced() {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

// Pulse envelope: fast attack, long settle, normalised to [0,1].
export function envelope(elapsed, dur = PULSE_MS) {
  if (elapsed < 0 || elapsed > dur) return 0;
  const t = elapsed / dur;
  const attack = 0.18;
  return t < attack ? t / attack : (1 - (t - attack) / (1 - attack)) ** 1.6;
}

// Barrel profile: 1 at the middle of the glass, 0 at its rim.
export function bulgeAt(v) {
  const d = (v - 0.5) * 2; // -1 .. 1
  return 1 - d * d;
}

// The cold start. Before the game appears, the tube says what it is you are about to watch —
// the page has no other moment where it can state plainly that the thing playing DOOM is the
// model. Typed out, phosphor on black, and it runs through the same warp and scanlines as the
// footage that follows.
const BOOT_WORD = 'TYPICAL';
const BOOT_LINES = [
  'open decision models · 1.7B and 4B',
  '',
  'state read once, cached · choice · yes/no · score · ∅',
  'now playing: DOOM E1M1, one sentence per tick',
];
const BOOT_CPS = 85; // characters a second for the lines under the word
const BOOT_WORD_MS = 1100; // the word rasterises in over this
const BOOT_HOLD = 1500; // after the last character, so there is time to read it
// The hand-over is the switch on an old set, not a cross-fade: the picture collapses to a
// scan line, the line flares, and the next picture opens back out of it.
const SW_COLLAPSE = 260;
const SW_FLASH = 110;
const SW_EXPAND = 340;
const FILL_CHAR = '·';
const FRONT_CHAR = '▓'; // the face of the word
const SIDE_CHAR = '▒'; // the sides of the extrusion

// Where the cold start is at a given moment: how much of the word has rasterised, how much of
// the text under it has been typed, and how far into the hand-over to the game we are.
export function bootState(elapsed, chars) {
  const word = Math.max(0, Math.min(1, elapsed / BOOT_WORD_MS));
  const typed = Math.min(chars, Math.max(0, Math.floor(((elapsed - BOOT_WORD_MS) / 1000) * BOOT_CPS)));
  const typedMs = BOOT_WORD_MS + (chars / BOOT_CPS) * 1000;
  const switchAt = typedMs + BOOT_HOLD;
  const t = elapsed - switchAt;
  let phase = 'text';
  let k = 0; // 0..1 within the phase
  if (t >= 0 && t < SW_COLLAPSE) {
    phase = 'collapse';
    k = t / SW_COLLAPSE;
  } else if (t >= SW_COLLAPSE && t < SW_COLLAPSE + SW_FLASH) {
    phase = 'flash';
    k = (t - SW_COLLAPSE) / SW_FLASH;
  } else if (t >= SW_COLLAPSE + SW_FLASH && t < SW_COLLAPSE + SW_FLASH + SW_EXPAND) {
    phase = 'expand';
    k = (t - SW_COLLAPSE - SW_FLASH) / SW_EXPAND;
  } else if (t >= SW_COLLAPSE + SW_FLASH + SW_EXPAND) {
    phase = 'done';
    k = 1;
  }
  return { word, typed, phase, k, done: phase === 'done', switchAt };
}

function canvas2d(w = 1, h = 1, opts) {
  const el = document.createElement('canvas');
  el.width = w;
  el.height = h;
  return { el, ctx: el.getContext('2d', opts) };
}

function noiseTile(size = 96) {
  const { el, ctx } = canvas2d(size, size);
  const img = ctx.createImageData(size, size);
  const fill = () => {
    const d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      const v = 120 + ((Math.random() * 70) | 0);
      d[i] = d[i + 1] = d[i + 2] = v;
      d[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  };
  fill();
  return { el, fill };
}

export function mountFilmFx(host, getSource, opts = {}) {
  if (!host) return { pulse() {}, stop() {} };
  const still = reduced();

  const layer = document.createElement('canvas');
  layer.className = 'film-fx';
  layer.setAttribute('aria-hidden', 'true');
  const film = opts.after;
  if (film?.parentNode === host) host.insertBefore(layer, film.nextSibling);
  else host.appendChild(layer);
  const ctx = layer.getContext('2d');
  // the iframe keeps running and keeps being readable; it just is not what you are looking at
  if (film) film.style.opacity = '0';

  const warp = canvas2d(); // the frame, warped, before the glass treatment
  const bloom = canvas2d(); // the lit parts only, at a quarter scale: blur cost is per pixel
  const grain = noiseTile();
  let scan = null;
  let grille = null;
  let vignette = null;
  let grainPat = null;

  let raf = 0;
  let last = 0;
  let stopped = false;
  let pulseAt = -1e9;
  let lastFilter = 'none'; // what the pulse is currently grading with, for verification
  let switching = null; // set during the hand-over, read when the buffer is composited
  let phase = 'boot'; // 'boot' | 'collapse' | 'flash' | 'expand' | 'live', exposed for verification
  const bootChars = BOOT_LINES.join('\n').length;
  // Every load, not once per session: a reload that skipped it read as the opening being broken.
  // It is short enough to sit through, and reduced motion still skips it outright.
  const skipBoot = still || opts.boot === false;
  let bootAt = skipBoot ? -1e9 : 0; // set on the first frame that has a source to draw
  let booted = skipBoot;
  if (skipBoot && opts.onReady) opts.onReady();

  function size(src) {
    const w = host.clientWidth;
    const h = host.clientHeight;
    if (!w || !h || !src?.width) return false;
    const dpr = Math.min(2, devicePixelRatio || 1);
    const W = Math.round(w * dpr);
    const H = Math.round(h * dpr);
    if (layer.width !== W || layer.height !== H) {
      layer.width = W;
      layer.height = H;
      layer.style.width = `${w}px`;
      layer.style.height = `${h}px`;
      // The source is 800x600; warping into a 2880-wide buffer just costs fill rate. Build the
      // picture at roughly source scale and let the final composite do the upscale.
      const ww = Math.min(W, 1024);
      warp.el.width = ww;
      warp.el.height = Math.round((H / W) * ww);
      bloom.el.width = Math.max(2, Math.round(W / 4));
      bloom.el.height = Math.max(2, Math.round(H / 4));
      scan = null;
      grille = null;
      vignette = null;
    }
    return true;
  }

  function patterns(dpr) {
    if (!scan) {
      const step = Math.max(3, Math.round(3 * dpr));
      const { el, ctx: c } = canvas2d(1, step);
      c.fillStyle = 'rgba(0,0,0,0.85)';
      c.fillRect(0, 0, 1, Math.max(1, Math.round(step / 3)));
      scan = ctx.createPattern(el, 'repeat');
    }
    if (!grille) {
      const step = Math.max(3, Math.round(3 * dpr));
      const { el, ctx: c } = canvas2d(step, 1);
      c.fillStyle = 'rgba(0,0,0,0.5)';
      c.fillRect(0, 0, 1, 1);
      grille = ctx.createPattern(el, 'repeat');
    }
    return { scan, grille };
  }

  // Strip-wise barrel warp. Each horizontal slice is drawn wider the closer it is to the middle of
  // the glass, which bows the verticals outwards, and shifted slightly away from the centre line,
  // which bows the horizontals. At this scale it is indistinguishable from a real lens warp and
  // costs a hundred drawImage calls instead of a per-pixel remap.
  function drawWarped(src, W, H, pulse) {
    const c = warp.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    if (c.globalAlpha === 1) c.clearRect(0, 0, W, H); // during the boot fade the text stays under
    c.imageSmoothingEnabled = false;
    c.filter = pulse > 0.01
      ? `contrast(${(1 + 0.22 * pulse).toFixed(3)}) saturate(${(1 + 0.16 * pulse).toFixed(3)})`
      : 'none';
    lastFilter = c.filter;
    const sh = src.height / STRIPS;
    const ampX = BULGE * W;
    const ampY = BULGE * H * 0.55;
    for (let i = 0; i < STRIPS; i++) {
      const v0 = i / STRIPS;
      const v1 = (i + 1) / STRIPS;
      // vertical bow: rows near the middle sit slightly further from the centre line
      const y0 = v0 * H + Math.sign(v0 - 0.5) * ampY * bulgeAt(v0) * 0.5;
      const y1 = v1 * H + Math.sign(v1 - 0.5) * ampY * bulgeAt(v1) * 0.5;
      // horizontal bow: the widest strips are the ones halfway down
      const over = ampX * bulgeAt((v0 + v1) / 2);
      c.drawImage(src, 0, i * sh, src.width, sh, -over, y0, W + over * 2, Math.max(1, y1 - y0) + 1);
    }
    c.filter = 'none';
  }

  // The wordmark is a solid: the letterforms are rasterised to a bitmap, every filled cell is
  // extruded into a box, and the faces that face open air become the model. It is then turned on
  // a slow turntable and z-buffered back down into the character grid, so the front of the word
  // and the sides of the extrusion get different characters. Geometry is built once; only the
  // projection runs per frame.
  let mesh = null;
  function buildMesh() {
    if (mesh) return mesh;
    const RES = 13; // bitmap rows — the whole solid is built from this, so keep it small
    const probe = canvas2d(8, 8).ctx;
    let size = RES;
    const track = (px) => px * 0.16;
    const widthAt = (px) => {
      probe.font = `700 ${px}px "Geist Mono", ui-monospace, monospace`;
      return [...BOOT_WORD].reduce((w, ch) => w + probe.measureText(ch).width, 0) + track(px) * (BOOT_WORD.length - 1);
    };
    const cols = Math.ceil(widthAt(size)) + 2;
    const { ctx: c } = canvas2d(cols, RES);
    c.fillStyle = '#000';
    c.fillRect(0, 0, cols, RES);
    c.font = `700 ${size}px "Geist Mono", ui-monospace, monospace`;
    c.textBaseline = 'middle';
    c.fillStyle = '#fff';
    let x = (cols - widthAt(size)) / 2;
    for (const ch of BOOT_WORD) {
      c.fillText(ch, x, RES / 2);
      x += c.measureText(ch).width + track(size);
    }
    const px = c.getImageData(0, 0, cols, RES).data;
    const on = [];
    for (let y = 0; y < RES; y++) {
      on[y] = new Uint8Array(cols);
      for (let cx = 0; cx < cols; cx++) on[y][cx] = px[(y * cols + cx) * 4] > 110 ? 1 : 0;
    }
    const halfD = 2.6; // extrusion depth, in bitmap cells
    const faces = [];
    const at = (yy, xx) => (yy < 0 || yy >= RES || xx < 0 || xx >= cols ? 0 : on[yy][xx]);
    for (let by = 0; by < RES; by++) {
      for (let bx = 0; bx < cols; bx++) {
        if (!on[by][bx]) continue;
        const x0 = bx - cols / 2;
        const y0 = RES / 2 - by;
        // front face, then a side face wherever this cell borders open air
        faces.push({ v: [[x0, y0, -halfD], [x0 + 1, y0, -halfD], [x0 + 1, y0 - 1, -halfD], [x0, y0 - 1, -halfD]], n: [0, 0, -1], t: 0 });
        if (!at(by - 1, bx)) faces.push({ v: [[x0, y0, halfD], [x0, y0, -halfD], [x0 + 1, y0, -halfD], [x0 + 1, y0, halfD]], n: [0, 1, 0], t: 1 });
        if (!at(by + 1, bx)) faces.push({ v: [[x0, y0 - 1, -halfD], [x0, y0 - 1, halfD], [x0 + 1, y0 - 1, halfD], [x0 + 1, y0 - 1, -halfD]], n: [0, -1, 0], t: 1 });
        if (!at(by, bx - 1)) faces.push({ v: [[x0, y0, -halfD], [x0, y0, halfD], [x0, y0 - 1, halfD], [x0, y0 - 1, -halfD]], n: [-1, 0, 0], t: 1 });
        if (!at(by, bx + 1)) faces.push({ v: [[x0 + 1, y0, halfD], [x0 + 1, y0, -halfD], [x0 + 1, y0 - 1, -halfD], [x0 + 1, y0 - 1, halfD]], n: [1, 0, 0], t: 1 });
      }
    }
    mesh = { faces, cols, rows: RES };
    return mesh;
  }

  // luminance + which kind of face won, per character cell
  let buf = null;
  function renderMesh(gCols, gRows, ry, scale, offY) {
    if (!buf || buf.cols !== gCols || buf.rows !== gRows) {
      buf = { cols: gCols, rows: gRows, lum: new Float32Array(gCols * gRows), face: new Uint8Array(gCols * gRows), z: new Float32Array(gCols * gRows) };
    }
    buf.lum.fill(0);
    buf.z.fill(-Infinity);
    const rx = -0.42; // a fixed tilt, so the turntable reads as a solid and not a flat card
    const cosX = Math.cos(rx), sinX = Math.sin(rx), cosY = Math.cos(ry), sinY = Math.sin(ry);
    const fov = 400;
    const camZ = 135; // gentler than the reference's 80: at this size it was collapsing the last letter
    const light = [0.6, 0.5, -0.62];
    const rot = ([x, y, z]) => {
      const y1 = y * cosX - z * sinX;
      const z1 = y * sinX + z * cosX;
      const x2 = x * cosY + z1 * sinY;
      const z2 = -x * sinY + z1 * cosY;
      return [x2, y1, z2];
    };
    const tri = (a, b, cc, lum, t) => {
      const minX = Math.max(0, Math.floor(Math.min(a[0], b[0], cc[0])));
      const maxX = Math.min(gCols - 1, Math.ceil(Math.max(a[0], b[0], cc[0])));
      const minY = Math.max(0, Math.floor(Math.min(a[1], b[1], cc[1])));
      const maxY = Math.min(gRows - 1, Math.ceil(Math.max(a[1], b[1], cc[1])));
      const dx01 = b[0] - a[0], dy01 = b[1] - a[1], dx02 = cc[0] - a[0], dy02 = cc[1] - a[1];
      const den = dx01 * dy02 - dx02 * dy01;
      if (Math.abs(den) < 1e-4) return;
      const inv = 1 / den;
      for (let py = minY; py <= maxY; py++) {
        for (let pxx = minX; pxx <= maxX; pxx++) {
          const dpx = pxx + 0.5 - a[0];
          const dpy = py + 0.5 - a[1];
          const u = (dpx * dy02 - dx02 * dpy) * inv;
          const v = (dx01 * dpy - dpx * dy01) * inv;
          if (u < 0 || v < 0 || u + v > 1) continue;
          const z = a[2] + u * (b[2] - a[2]) + v * (cc[2] - a[2]);
          const n = py * gCols + pxx;
          if (z <= buf.z[n]) continue;
          buf.z[n] = z;
          buf.lum[n] = lum;
          buf.face[n] = t;
        }
      }
    };
    for (const f of mesh.faces) {
      const n = rot(f.n);
      if (n[2] > 0.05) continue; // back faces: the camera looks down -z
      const lum = Math.max(0.15, -(n[0] * light[0] + n[1] * light[1] + n[2] * light[2]) * 0.7 + 0.35);
      const p = f.v.map((v) => {
        const r = rot([v[0] * scale, v[1] * scale, v[2] * scale]);
        const pz = Math.max(1, camZ - r[2]);
        return [(r[0] * fov) / pz + gCols / 2, (-r[1] * fov) / pz + gRows / 2 + offY, r[2]];
      });
      tri(p[0], p[1], p[2], lum, f.t);
      tri(p[0], p[2], p[3], lum, f.t);
    }
    return buf;
  }

  function drawBoot(W, H, b) {
    const c = warp.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.fillStyle = '#06080a';
    c.fillRect(0, 0, W, H);
    const cell = Math.max(7, Math.round(H * 0.0155));
    const cw = cell * 0.62;
    const gCols = Math.floor(W / cw);
    const gRows = Math.floor(H / cell);
    buildMesh();
    // It turns in from three-quarters, then keeps rocking on a slow turntable: held face-on the
    // extrusion disappears and the word reads as flat type, which is the whole point of building
    // it as a solid.
    const spin = (1 - b.word) ** 2;
    const t = performance.now() / 1000;
    const ry = -1.25 * spin + (1 - spin) * -0.34 * Math.cos(t * 0.75);
    const scale = 0.27 + 0.04 * b.word;
    const grid = renderMesh(gCols, gRows, ry, scale, -gRows * 0.06);
    c.font = `${cell}px "Geist Mono", ui-monospace, monospace`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    for (let y = 0; y < gRows; y++) {
      for (let x = 0; x < gCols; x++) {
        const n = y * gCols + x;
        const cx = (x + 0.5) * cw;
        const cy = (y + 0.5) * cell;
        const lum = grid.lum[n];
        if (lum > 0.01) {
          const front = grid.face[n] === 0;
          const a = front ? 0.7 + 0.3 * lum : 0.35 + 0.45 * lum;
          c.fillStyle = `rgba(190,255,215,${a.toFixed(2)})`;
          c.fillText(front ? FRONT_CHAR : SIDE_CHAR, cx, cy);
        } else {
          c.fillStyle = 'rgba(120,200,160,0.085)';
          c.fillText(FILL_CHAR, cx, cy);
        }
      }
    }
    // The byline sits under the solid, small: the wordmark is the sculpture, this is the maker's
    // mark. It arrives once the word has landed, so it does not compete with it.
    if (b.word > 0.85) {
      let lo = gRows;
      let hi = 0;
      let left = gCols;
      let right = 0;
      for (let y = 0; y < gRows; y++) {
        for (let x = 0; x < gCols; x++) {
          if (grid.lum[y * gCols + x] > 0.01) {
            if (y < lo) lo = y;
            if (y > hi) hi = y;
            if (x < left) left = x;
            if (x > right) right = x;
          }
        }
      }
      if (hi >= lo) {
        const bySize = Math.max(10, Math.round(cell * 1.15));
        c.font = `${bySize}px "Geist Mono", ui-monospace, monospace`;
        c.textAlign = 'center';
        c.textBaseline = 'top';
        c.fillStyle = `rgba(190,255,215,${(0.5 * Math.min(1, (b.word - 0.85) / 0.15)).toFixed(2)})`;
        c.fillText('by OZ LABS', ((left + right + 1) / 2) * cw, (hi + 2.2) * cell);
      }
    }

    // the lines under it, typed
    const size = Math.max(11, Math.round(H * 0.021));
    c.font = `${size}px "Geist Mono", ui-monospace, monospace`;
    c.textAlign = 'left';
    c.textBaseline = 'top';
    const x0 = Math.round(W * 0.085);
    let y = Math.round(H * 0.72);
    let left = b.typed;
    for (const line of BOOT_LINES) {
      if (left <= 0) break;
      const shown = line.slice(0, left);
      left -= line.length + 1;
      c.fillStyle = 'rgba(190,255,215,0.9)';
      if (shown) c.fillText(shown, x0, y);
      if (left <= 0) c.fillRect(x0 + c.measureText(shown).width + 4, y + 2, size * 0.5, size);
      y += Math.round(size * 1.7);
    }
  }

  function draw(src, now) {
    const W = layer.width;
    const H = layer.height;
    const dpr = Math.min(2, devicePixelRatio || 1);
    const pulse = envelope(now - pulseAt);
    if (!booted) {
      if (bootAt === 0) bootAt = now;
      // The reveal waits for the game to actually be in the level: without this the cold start
      // handed over to DOOM's title screen, which is not what the text just promised.
      // Hold the text on screen until the game is genuinely settled in a level. Without this the
      // switch happened over whatever DOOM was doing at that instant — a glimpse of play, then
      // its own title screen coming back.
      const { switchAt } = bootState(0, bootChars);
      if (opts.ready && !opts.ready() && now - bootAt > switchAt) bootAt = now - switchAt;
      const b = bootState(now - bootAt, bootChars);
      phase = b.done ? 'live' : b.phase === 'text' ? 'boot' : b.phase;
      if (b.done) {
        booted = true;
        switching = null;
        opts.onReady?.();
      } else if (b.phase === 'expand') {
        // the game is what opens back out of the line
        drawWarped(src, warp.el.width, warp.el.height, pulse);
        switching = b;
      } else {
        drawBoot(warp.el.width, warp.el.height, b);
        switching = b.phase === 'text' ? null : b;
      }
    }
    if (booted) {
      // Hold the last good frame whenever the game is not in a level. DOOM's own demo sequencer
      // can pull a running level back to the attract screen (typical_new_game only defers an
      // init; it does not clear demoplayback/advancedemo), and the recovery takes a beat. Since
      // the picture on screen is ours, we simply keep showing the last frame of play instead of
      // letting the title screen appear on the hero.
      if (!opts.live || opts.live()) drawWarped(src, warp.el.width, warp.el.height, pulse);
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    // No inset and no bezel: the tube fills the panel. The hero's own rounded corners come from
    // the stage (it clips, and the scroll inset rounds it), so the glass does not need its own.
    ctx.imageSmoothingEnabled = false;
    if (switching) {
      // collapse to the line, flare, then open back out of it
      const ease = (v) => 1 - (1 - v) ** 2;
      const h = switching.phase === 'collapse' ? 1 - ease(switching.k)
        : switching.phase === 'flash' ? 0
        : ease(switching.k);
      const mid = H / 2;
      if (h > 0.001) ctx.drawImage(warp.el, 0, 0, warp.el.width, warp.el.height, 0, mid - (H * h) / 2, W, H * h);
      const lineA = switching.phase === 'flash' ? 1 : switching.phase === 'collapse' ? switching.k : Math.max(0, 1 - switching.k * 2.2);
      if (lineA > 0.01) {
        const lh = Math.max(2, H * 0.004 * (1 + (switching.phase === 'flash' ? 1.6 : 0)));
        const g2 = ctx.createLinearGradient(0, mid - lh * 6, 0, mid + lh * 6);
        g2.addColorStop(0, 'rgba(200,255,225,0)');
        g2.addColorStop(0.5, `rgba(220,255,235,${(0.75 * lineA).toFixed(3)})`);
        g2.addColorStop(1, 'rgba(200,255,225,0)');
        ctx.fillStyle = g2;
        ctx.fillRect(0, mid - lh * 6, W, lh * 12);
        ctx.fillStyle = `rgba(240,255,245,${lineA.toFixed(3)})`;
        ctx.fillRect(0, mid - lh / 2, W, lh);
      }
    } else {
      ctx.drawImage(warp.el, 0, 0, W, H);
    }

    // bloom, from the picture itself: pushed through a curve so only what was actually lit
    // survives, blurred at a quarter scale and added back. Blurring at full resolution cost three
    // quarters of the frame budget for a halo that is soft by definition.
    const bw = bloom.el.width;
    const bh = bloom.el.height;
    const bc = bloom.ctx;
    bc.setTransform(1, 0, 0, 1, 0, 0);
    bc.clearRect(0, 0, bw, bh);
    bc.filter = 'brightness(1.45) contrast(2.2) blur(2px)';
    bc.drawImage(warp.el, 0, 0, warp.el.width, warp.el.height, 0, 0, bw, bh);
    bc.filter = 'none';
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.imageSmoothingEnabled = true;
    ctx.globalAlpha = 0.34 + 0.3 * pulse;
    ctx.drawImage(bloom.el, 0, 0, W, H);
    ctx.globalAlpha = 0.16 + 0.2 * pulse;
    ctx.filter = 'blur(6px)';
    ctx.drawImage(bloom.el, -W * 0.01, -H * 0.01, W * 1.02, H * 1.02);
    ctx.restore();
    ctx.filter = 'none';

    const pat = patterns(dpr);
    ctx.save();
    ctx.globalCompositeOperation = 'multiply';
    ctx.globalAlpha = 0.5 + 0.18 * pulse;
    ctx.fillStyle = pat.scan;
    ctx.fillRect(0, 0, W, H);
    ctx.globalAlpha = 0.16;
    ctx.fillStyle = pat.grille;
    ctx.fillRect(0, 0, W, H);
    ctx.restore();

    if (!vignette) {
      vignette = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.22, W / 2, H / 2, Math.max(W, H) * 0.72);
      vignette.addColorStop(0, 'rgba(0,0,0,0)');
      vignette.addColorStop(0.7, 'rgba(0,0,0,0.2)');
      vignette.addColorStop(1, 'rgba(0,0,0,0.6)');
    }
    ctx.fillStyle = vignette;
    ctx.fillRect(0, 0, W, H);

    if (!still) {
      if ((now / 120) % 2 < 1) grain.fill();
      if (!grainPat) grainPat = ctx.createPattern(grain.el, 'repeat');
      ctx.save();
      ctx.globalCompositeOperation = 'overlay';
      ctx.globalAlpha = 0.07;
      ctx.translate(-(Math.random() * 90) | 0, -(Math.random() * 90) | 0);
      ctx.fillStyle = grainPat;
      ctx.fillRect(0, 0, W + 96, H + 96);
      ctx.restore();
    }

    ctx.restore();
  }

  function frame(t) {
    if (stopped) return;
    raf = requestAnimationFrame(frame);
    if (t - last < 1000 / FPS) return;
    last = t;
    const src = getSource();
    if (!src || !size(src)) return;
    try {
      draw(src, t);
    } catch {
      stop();
      layer.remove();
      if (film) film.style.opacity = ''; // fall back to the plain film rather than a blank hero
    }
  }

  function stop() {
    stopped = true;
    cancelAnimationFrame(raf);
    if (film) film.style.opacity = '';
  }

  raf = requestAnimationFrame(frame);

  return {
    pulse() {
      pulseAt = performance.now();
    },
    grade: () => lastFilter,
    phase: () => (booted ? 'live' : phase),
    stop,
  };
}

export function selfTest() {
  const chars = BOOT_LINES.join('\n').length;
  console.assert(bootState(0, chars).typed === 0, 'nothing is typed at zero');
  console.assert(bootState(BOOT_WORD_MS + 1000, chars).typed === BOOT_CPS, 'typing runs at the stated rate');
  console.assert(bootState(0, chars).word === 0 && bootState(BOOT_WORD_MS, chars).word === 1, 'the word rasterises in over its own window');
  const at = bootState(0, chars).switchAt;
  console.assert(bootState(at - 1, chars).phase === 'text', 'the text holds until the switch');
  console.assert(bootState(at + 10, chars).phase === 'collapse', 'then the picture collapses');
  console.assert(bootState(at + SW_COLLAPSE + 10, chars).phase === 'flash', 'the line flares');
  console.assert(bootState(at + SW_COLLAPSE + SW_FLASH + 10, chars).phase === 'expand', 'and the game opens out of it');
  console.assert(bootState(at + SW_COLLAPSE + SW_FLASH + SW_EXPAND + 1, chars).done, 'then it is over');
  console.assert(bootState(1e6, chars).done, 'the boot always finishes');
  console.assert(!bootState((chars / BOOT_CPS) * 1000 + 100, chars).done, 'and holds before it fades');
  console.assert(envelope(-1) === 0 && envelope(PULSE_MS + 1) === 0, 'the pulse is silent outside its window');
  console.assert(Math.abs(envelope(PULSE_MS * 0.18) - 1) < 1e-6, 'the pulse peaks at the end of the attack');
  console.assert(envelope(PULSE_MS * 0.6) < envelope(PULSE_MS * 0.3), 'and settles after it');
  console.assert(bulgeAt(0.5) === 1 && bulgeAt(0) === 0 && bulgeAt(1) === 0, 'the glass swells in the middle and is flat at the rim');
  console.assert(Math.abs(bulgeAt(0.25) - 0.75) < 1e-9, 'and eases between');
  console.log('filmfx.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}
