// filmfx.js — the hero film, seen through the decision machine.
//
// The film itself stays where it is: a sharp iframe underneath. Everything here is additive, drawn
// into one overlay canvas that sits over the film and under the page's own chrome, so navigation,
// type and cards are never touched.
//
// What it adds, in draw order:
//   1. a contrast lift, only while a decision is being processed
//   2. RGB separation — red and blue fringes, and only where the frame has a hard edge
//   3. phosphor bloom, weighted by luminance so a lit lamp halates and a dark corridor does not
//   4. patches: small sparse regions that briefly resolve into tiny characters or edge contours
//      and then reconstruct into the pixels underneath
//   5. fine scanlines
//   6. a little animated grain
//
// The masks (edges, highlights) are computed at half the film's resolution in one pass over an
// ImageData and composited back up. Full resolution would be ~500k iterations a frame for no
// visible gain: the separation is a pixel wide and the bloom is blurred anyway.
//
//   const fx = mountFilmFx(host, getSource, { after });
//   fx.pulse();   // a decision landed: 300-500ms of heightened contrast, bloom and separation
//   fx.stop();

const FPS = 30; // the film ticks at ~2.5 decisions/s; 30 is plenty and leaves the CPU alone
const SCALE = 0.4; // mask resolution relative to the source frame
const EDGE_T = 0.3; // gradient magnitude above which a pixel counts as an edge
const HOT_T = 0.62; // luminance above which a pixel blooms
const PULSE_MS = 420; // inside the 300-500ms the brief asks for

const GLYPHS = ' .:-=+*#%@';

function reduced() {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

// Pulse envelope: a fast attack and a long settle, normalised to [0,1]. Exported for the test.
export function envelope(elapsed, dur = PULSE_MS) {
  if (elapsed < 0 || elapsed > dur) return 0;
  const t = elapsed / dur;
  const attack = 0.18;
  return t < attack ? t / attack : (1 - (t - attack) / (1 - attack)) ** 1.6;
}

// Patch envelope: fade in, hold, fade out.
export function patchAlpha(elapsed, dur) {
  if (elapsed < 0 || elapsed > dur) return 0;
  const t = elapsed / dur;
  if (t < 0.25) return t / 0.25;
  if (t > 0.65) return Math.max(0, 1 - (t - 0.65) / 0.35);
  return 1;
}

function canvas2d(w = 1, h = 1, opts) {
  const el = document.createElement('canvas');
  el.width = w;
  el.height = h;
  return { el, ctx: el.getContext('2d', opts) };
}

// A tile of monochrome noise, redrawn every few frames and scrolled between redraws so the grain
// moves without costing a new tile each time.
function noiseTile(size = 96) {
  const { el, ctx } = canvas2d(size, size);
  const img = ctx.createImageData(size, size);
  const fill = () => {
    const d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      const v = 118 + ((Math.random() * 74) | 0);
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
  if (opts.after?.parentNode === host) host.insertBefore(layer, opts.after.nextSibling);
  else host.appendChild(layer);
  const ctx = layer.getContext('2d');

  const work = canvas2d(1, 1, { willReadFrequently: true }); // source at mask resolution
  const fringe = canvas2d(); // red/blue edge fringes
  const hot = canvas2d(); // highlights only, for the bloom
  const contour = canvas2d(); // edge contours, for the patches that resolve into lines
  const grain = noiseTile();
  let lines = null; // cached scanline pattern, rebuilt on resize

  let raf = 0;
  let last = 0;
  let stopped = false;
  let mw = 0;
  let mh = 0;
  let pulseAt = -1e9;
  let nextPatchAt = performance.now() + 2500;
  const patches = [];

  function size(src) {
    const w = host.clientWidth;
    const h = host.clientHeight;
    if (!w || !h || !src?.width) return false;
    const dpr = Math.min(2, devicePixelRatio || 1);
    if (layer.width !== Math.round(w * dpr) || layer.height !== Math.round(h * dpr)) {
      layer.width = Math.round(w * dpr);
      layer.height = Math.round(h * dpr);
      layer.style.width = `${w}px`;
      layer.style.height = `${h}px`;
      lines = null;
    }
    const nw = Math.max(2, Math.round(src.width * SCALE));
    const nh = Math.max(2, Math.round(src.height * SCALE));
    if (nw !== mw || nh !== mh) {
      mw = nw;
      mh = nh;
      [work, fringe, hot, contour].forEach((c) => {
        c.el.width = mw;
        c.el.height = mh;
      });
    }
    return true;
  }

  // One pass over the frame: luminance, Sobel magnitude and sign, and the three masks that come
  // out of them. Writing three ImageDatas in the same loop keeps this to a single read.
  function masks(src, pulse) {
    work.ctx.drawImage(src, 0, 0, mw, mh);
    const px = work.ctx.getImageData(0, 0, mw, mh).data;
    const fImg = fringe.ctx.createImageData(mw, mh);
    const hImg = hot.ctx.createImageData(mw, mh);
    const cImg = contour.ctx.createImageData(mw, mh);
    const f = fImg.data;
    const hd = hImg.data;
    const cd = cImg.data;
    const lum = new Float32Array(mw * mh);
    for (let i = 0, n = 0; n < lum.length; i += 4, n++) {
      lum[n] = (px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114) / 255;
    }
    // Interior pixels only, indexed arithmetically: the clamped accessor this used to call six
    // times per pixel was most of the frame budget, and a one-pixel border carries no edges worth
    // fringing. The highlight mask still covers the whole frame (its loop has no neighbours).
    for (let n = 0; n < lum.length; n++) {
      const l0 = lum[n];
      if (l0 > HOT_T) {
        const i0 = n * 4;
        const a = ((l0 - HOT_T) / (1 - HOT_T)) ** 0.9;
        hd[i0] = px[i0];
        hd[i0 + 1] = px[i0 + 1];
        hd[i0 + 2] = px[i0 + 2];
        hd[i0 + 3] = Math.round(a * 255);
      }
    }
    for (let y = 1; y < mh - 1; y++) {
      const row = y * mw;
      for (let x = 1; x < mw - 1; x++) {
        const n = row + x;
        const i = n * 4;
        const tl = lum[n - mw - 1], tc = lum[n - mw], tr = lum[n - mw + 1];
        const ml = lum[n - 1], mr = lum[n + 1];
        const bl = lum[n + mw - 1], bc = lum[n + mw], br = lum[n + mw + 1];
        const gx = tr + 2 * mr + br - tl - 2 * ml - bl;
        const gy = bl + 2 * bc + br - tl - 2 * tc - tr;
        const mag = Math.abs(gx) + Math.abs(gy); // cheaper than hypot and the threshold absorbs the difference
        if (mag > EDGE_T) {
          const a = Math.min(1, (mag - EDGE_T) * 1.5);
          // the side the edge falls on decides the colour: a horizontal step gets a red fringe on
          // one flank and a blue one on the other, which is what lens/electron misconvergence does
          const warm = gx > 0;
          // dim and desaturated: a hint of misconvergence, not a colour-separated glitch
          f[i] = warm ? 150 : 46;
          f[i + 1] = 40;
          f[i + 2] = warm ? 46 : 150;
          f[i + 3] = Math.round(a * 255 * (0.16 + 0.26 * pulse));
          cd[i] = cd[i + 1] = cd[i + 2] = 240;
          cd[i + 3] = Math.round(Math.min(1, mag * 1.2) * 255);
        }
      }
    }
    fringe.ctx.putImageData(fImg, 0, 0);
    hot.ctx.putImageData(hImg, 0, 0);
    contour.ctx.putImageData(cImg, 0, 0);
    return { px };
  }

  function scanlines(W, H) {
    if (lines) return lines;
    const step = Math.max(2, Math.round(2 * Math.min(2, devicePixelRatio || 1)));
    const { el, ctx: c } = canvas2d(1, step);
    c.fillStyle = 'rgba(0,0,0,0.1)';
    c.fillRect(0, 0, 1, 1);
    lines = ctx.createPattern(el, 'repeat');
    void W;
    void H;
    return lines;
  }

  // A patch: a small region that resolves into characters, or into its own edge contours, and
  // then reconstructs. Positions avoid the bottom-left copy card and the bottom-right HUD.
  function spawnPatch(now, W, H) {
    const kind = Math.random() < 0.55 ? 'glyphs' : 'contour';
    const w = (0.07 + Math.random() * 0.1) * W;
    const h = w * (0.55 + Math.random() * 0.5);
    const x = Math.random() * (W - w);
    const y = Math.random() * (H * 0.62);
    patches.push({ kind, x, y, w, h, born: now, dur: 900 + Math.random() * 700 });
  }

  function drawGlyphPatch(p, a, px) {
    const cell = 7 * Math.min(2, devicePixelRatio || 1);
    const cols = Math.max(1, Math.round(p.w / cell));
    const rows = Math.max(1, Math.round(p.h / (cell * 1.6)));
    ctx.save();
    ctx.globalAlpha = a * 0.85;
    ctx.fillStyle = 'rgba(8,8,9,0.72)'; // the pixels step back so the characters can be read
    ctx.fillRect(p.x, p.y, p.w, p.h);
    ctx.font = `${(p.h / rows).toFixed(1)}px "Geist Mono", ui-monospace, monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = 'rgba(240,238,235,0.92)';
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const sx = Math.min(mw - 1, Math.round(((p.x + (c + 0.5) * (p.w / cols)) / layer.width) * mw));
        const sy = Math.min(mh - 1, Math.round(((p.y + (r + 0.5) * (p.h / rows)) / layer.height) * mh));
        const i = (sy * mw + sx) * 4;
        const l = (px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114) / 255;
        const g = GLYPHS[Math.min(GLYPHS.length - 1, Math.floor(l ** 0.8 * GLYPHS.length))];
        if (g === ' ') continue;
        ctx.fillText(g, p.x + (c + 0.5) * (p.w / cols), p.y + (r + 0.5) * (p.h / rows));
      }
    }
    ctx.restore();
  }

  function drawContourPatch(p, a) {
    ctx.save();
    ctx.globalAlpha = a * 0.5;
    ctx.fillStyle = 'rgba(8,8,9,0.5)';
    ctx.fillRect(p.x, p.y, p.w, p.h);
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = a * 0.75;
    const sx = (p.x / layer.width) * mw;
    const sy = (p.y / layer.height) * mh;
    const sw = (p.w / layer.width) * mw;
    const sh = (p.h / layer.height) * mh;
    ctx.drawImage(contour.el, sx, sy, sw, sh, p.x, p.y, p.w, p.h);
    ctx.restore();
  }

  function draw(src, now) {
    const W = layer.width;
    const H = layer.height;
    const pulse = envelope(now - pulseAt);
    const { px } = masks(src, pulse);

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // 1. contrast, only while processing: the frame blended over itself through a contrast curve
    if (pulse > 0.01) {
      ctx.save();
      ctx.globalAlpha = 0.5 * pulse;
      ctx.filter = `contrast(${(1 + 0.22 * pulse).toFixed(3)}) saturate(${(1 + 0.1 * pulse).toFixed(3)})`;
      ctx.drawImage(src, 0, 0, W, H);
      ctx.restore();
      ctx.filter = 'none';
    }

    // 2. RGB separation at edges only
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = 0.3 + 0.3 * pulse;
    const dx = (0.7 + 1.3 * pulse) * (W / mw) * 0.5;
    ctx.drawImage(fringe.el, -dx, 0, W, H);
    ctx.globalAlpha = 0.22 + 0.26 * pulse;
    ctx.drawImage(fringe.el, dx, 0, W, H);
    ctx.restore();

    // 3. phosphor bloom
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.imageSmoothingEnabled = true;
    ctx.globalAlpha = 0.34 + 0.4 * pulse;
    ctx.filter = 'blur(7px)';
    ctx.drawImage(hot.el, 0, 0, W, H);
    ctx.globalAlpha = 0.2 + 0.28 * pulse;
    ctx.filter = 'blur(26px)';
    ctx.drawImage(hot.el, 0, 0, W, H);
    ctx.restore();
    ctx.filter = 'none';

    // 4. patches
    if (!still) {
      if (now > nextPatchAt && patches.length < 2) {
        spawnPatch(now, W, H);
        nextPatchAt = now + 2600 + Math.random() * 4200;
      }
      for (let i = patches.length - 1; i >= 0; i--) {
        const p = patches[i];
        const a = patchAlpha(now - p.born, p.dur);
        if (a <= 0) {
          patches.splice(i, 1);
          continue;
        }
        if (p.kind === 'glyphs') drawGlyphPatch(p, a, px);
        else drawContourPatch(p, a);
      }
    }

    // 5. scanlines
    ctx.save();
    ctx.globalCompositeOperation = 'multiply';
    ctx.fillStyle = scanlines(W, H);
    ctx.fillRect(0, 0, W, H);
    ctx.restore();

    // 6. grain
    if (!still) {
      if ((now / 90) % 2 < 1) grain.fill();
      ctx.save();
      ctx.globalCompositeOperation = 'overlay';
      ctx.globalAlpha = 0.055 + 0.03 * pulse;
      const p = ctx.createPattern(grain.el, 'repeat');
      ctx.translate(-(Math.random() * 90) | 0, -(Math.random() * 90) | 0);
      ctx.fillStyle = p;
      ctx.fillRect(0, 0, W + 96, H + 96);
      ctx.restore();
    }
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
    }
  }

  function stop() {
    stopped = true;
    cancelAnimationFrame(raf);
  }

  raf = requestAnimationFrame(frame);

  return {
    pulse() {
      pulseAt = performance.now();
    },
    stop,
  };
}

export function selfTest() {
  console.assert(envelope(-1) === 0 && envelope(PULSE_MS + 1) === 0, 'the pulse is silent outside its window');
  console.assert(Math.abs(envelope(PULSE_MS * 0.18) - 1) < 1e-6, 'the pulse peaks at the end of the attack');
  console.assert(envelope(PULSE_MS * 0.6) < envelope(PULSE_MS * 0.3), 'and settles after it');
  console.assert(patchAlpha(0, 1000) === 0 && patchAlpha(1000, 1000) === 0, 'a patch starts and ends invisible');
  console.assert(patchAlpha(500, 1000) === 1, 'and is fully resolved in the middle');
  console.log('filmfx.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}
