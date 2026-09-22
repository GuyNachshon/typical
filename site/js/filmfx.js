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
const BOOT_CPS = 60; // characters a second for the lines under the word
const BOOT_WORD_MS = 1100; // the word rasterises in over this
const BOOT_HOLD = 1000; // after the last character, before the game fades up
const BOOT_FADE = 700;
const FILL_CHAR = '·';
const ON_CHAR = '█';
const MID_CHAR = '▓';
const EDGE_CHAR = '▒';
const BOOT_KEY = 'typical_boot';

// Where the cold start is at a given moment: how much of the word has rasterised, how much of
// the text under it has been typed, and how far into the hand-over to the game we are.
export function bootState(elapsed, chars) {
  const word = Math.max(0, Math.min(1, elapsed / BOOT_WORD_MS));
  const typed = Math.min(chars, Math.max(0, Math.floor(((elapsed - BOOT_WORD_MS) / 1000) * BOOT_CPS)));
  const typedMs = BOOT_WORD_MS + (chars / BOOT_CPS) * 1000;
  const fade = Math.max(0, Math.min(1, (elapsed - typedMs - BOOT_HOLD) / BOOT_FADE));
  return { word, typed, fade, done: fade >= 1, typedMs };
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
  const bootChars = BOOT_LINES.join('\n').length;
  const skipBoot = still || opts.boot === false || (typeof sessionStorage !== 'undefined' && sessionStorage.getItem(BOOT_KEY));
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

  // The word, rasterised into a character grid: the letterforms are drawn once into a small
  // offscreen canvas, then read back so each cell knows whether it is inside a letter, on its
  // edge, or in the field around it. Rebuilt only when the grid changes size.
  let raster = null;
  function buildRaster(cols, rows) {
    if (raster && raster.cols === cols && raster.rows === rows) return raster;
    const { el, ctx: c } = canvas2d(cols, rows);
    c.fillStyle = '#000';
    c.fillRect(0, 0, cols, rows);
    let size = rows * 0.92;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillStyle = '#fff';
    // shrink to fit the width, letter-spaced by hand since canvas has no tracking
    const spacing = () => size * 0.14;
    const widthAt = (px) => {
      c.font = `700 ${px}px "Geist Mono", ui-monospace, monospace`;
      return [...BOOT_WORD].reduce((w, ch) => w + c.measureText(ch).width, 0) + spacing() * (BOOT_WORD.length - 1);
    };
    while (size > 4 && widthAt(size) > cols * 0.86) size *= 0.94;
    c.font = `700 ${size}px "Geist Mono", ui-monospace, monospace`;
    const total = widthAt(size);
    let x = (cols - total) / 2;
    for (const ch of BOOT_WORD) {
      const w = c.measureText(ch).width;
      c.fillText(ch, x + w / 2, rows / 2);
      x += w + spacing();
    }
    const px = c.getImageData(0, 0, cols, rows).data;
    const on = new Uint8Array(cols * rows);
    for (let i = 0, n = 0; n < on.length; i += 4, n++) on[n] = px[i] > 110 ? 1 : 0;
    // edge cells: inside the letter but next to something that is not
    const edge = new Uint8Array(cols * rows);
    for (let y = 0; y < rows; y++) {
      for (let cx = 0; cx < cols; cx++) {
        const n = y * cols + cx;
        if (!on[n]) continue;
        const nb = (dx, dy) => {
          const yy = y + dy;
          const xx = cx + dx;
          return yy < 0 || yy >= rows || xx < 0 || xx >= cols ? 0 : on[yy * cols + xx];
        };
        if (!nb(1, 0) || !nb(-1, 0) || !nb(0, 1) || !nb(0, -1)) edge[n] = 1;
      }
    }
    raster = { cols, rows, on, edge, el };
    return raster;
  }

  function drawBoot(W, H, b) {
    const c = warp.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.fillStyle = '#06080a';
    c.fillRect(0, 0, W, H);
    const cell = Math.max(6, Math.round(H * 0.0135));
    const cols = Math.floor(W / (cell * 0.62));
    const rows = Math.round((H * 0.26) / cell);
    const r = buildRaster(cols, rows);
    c.font = `${cell}px "Geist Mono", ui-monospace, monospace`;
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    const cw = W / cols;
    const top = Math.round(H * 0.16);
    // the field fills column by column, so the word arrives left to right
    const front = b.word * (cols + 8);
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const n = y * cols + x;
        const here = front - x;
        if (here <= 0) continue;
        const cx = (x + 0.5) * cw;
        const cy = top + (y + 0.5) * cell;
        if (r.on[n]) {
          // the leading column burns brighter as it lands, then settles
          const fresh = Math.max(0, 1 - here / 7);
          c.fillStyle = `rgba(${175 + 60 * fresh | 0},255,${205 + 40 * fresh | 0},${(0.8 + 0.2 * fresh).toFixed(2)})`;
          // a hint of texture across the letterform: solid in the middle, lighter at the edge
          const ch = r.edge[n] ? EDGE_CHAR : ((x + y) % 5 === 0 ? MID_CHAR : ON_CHAR);
          c.fillText(ch, cx, cy);
        } else {
          c.fillStyle = 'rgba(120,200,160,0.1)';
          c.fillText(FILL_CHAR, cx, cy);
        }
      }
    }
    // the lines under it, typed
    const size = Math.max(11, Math.round(H * 0.022));
    c.font = `${size}px "Geist Mono", ui-monospace, monospace`;
    c.textAlign = 'left';
    c.textBaseline = 'top';
    c.fillStyle = 'rgba(175,255,205,0.88)';
    const x0 = Math.round(W * 0.5 - (cols * cw * 0.43) / 2);
    let y = top + rows * cell + Math.round(H * 0.06);
    let left = b.typed;
    for (const line of BOOT_LINES) {
      if (left <= 0) break;
      const shown = line.slice(0, left);
      left -= line.length + 1;
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
      const { typedMs } = bootState(0, bootChars);
      if (opts.ready && !opts.ready() && now - bootAt > typedMs + BOOT_HOLD * 0.6) {
        bootAt = now - typedMs - BOOT_HOLD * 0.6;
      }
      const b = bootState(now - bootAt, bootChars);
      if (b.done) {
        booted = true;
        try { sessionStorage.setItem(BOOT_KEY, '1'); } catch {}
        opts.onReady?.();
      } else {
        drawBoot(warp.el.width, warp.el.height, b);
        if (b.fade > 0) {
          // the game fades up through the boot screen rather than cutting to it
          warp.ctx.globalAlpha = b.fade;
          drawWarped(src, warp.el.width, warp.el.height, pulse);
          warp.ctx.globalAlpha = 1;
        }
      }
    }
    if (booted) drawWarped(src, warp.el.width, warp.el.height, pulse);

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    // No inset and no bezel: the tube fills the panel. The hero's own rounded corners come from
    // the stage (it clips, and the scroll inset rounds it), so the glass does not need its own.
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(warp.el, 0, 0, W, H);

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
    stop,
  };
}

export function selfTest() {
  const chars = BOOT_LINES.join('\n').length;
  console.assert(bootState(0, chars).typed === 0, 'nothing is typed at zero');
  console.assert(bootState(BOOT_WORD_MS + 1000, chars).typed === BOOT_CPS, 'typing runs at the stated rate');
  console.assert(bootState(0, chars).word === 0 && bootState(BOOT_WORD_MS, chars).word === 1, 'the word rasterises in over its own window');
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
