// ascii.js — the hero film, rendered as characters.
//
// The argument the whole site makes is that this model never sees the picture: it reads a
// sentence and returns probabilities. So the film runs as a character field by default, and the
// real frame is underneath — hover (or focus, or tap) and the glyphs dissolve to reveal it.
// The overlay is what the page claims the model works from; the pixels are what it doesn't get.
//
//   mountAscii(host, getSource, opts) -> { stop() }
//
// host       element the overlay is absolutely positioned inside (needs position: relative);
//            its mask is written by this module, so don't set mask-image on .ascii-layer in CSS
// getSource  () => HTMLCanvasElement | null, re-read every frame (the iframe may still be booting)
// opts.cols  character columns (default 150); rows follow from the source aspect ratio
// opts.fps   sample rate (default 15; the film ticks at 2.5 decisions/s, glyph noise above ~15
//            reads as static rather than motion)

// Glyph set, dark to light. Not a pure density ramp: the reference look (dense terminal
// transcription of a frame) mixes letters, brackets and digits, which gives the field texture at
// small sizes where a ramp of #%@ turns into a flat grey block.
const BAYER = [0.25, 0.75, 1.0, 0.5]; // 2x2 ordered-dither thresholds
// 4x4 Bayer matrix for the 1-bit treatment: ordered dithering keeps structure at one bit per
// cell far better than a threshold, which is why every 1980s printer used it.
const BAYER4 = [
  0, 8, 2, 10,
  12, 4, 14, 6,
  3, 11, 1, 9,
  15, 7, 13, 5,
].map((v) => (v + 0.5) / 16);
const FLOOR = 0.2;
const HOT = 0.78; // luminance above which a dithered cell blooms // below this share of the frame's own range, a cell stays blank
const RAMP = [
  ' ', ' ', '.', ',', ':', ';', 'i', 'l', '!', '|', '/', '\\', '1', 'I', '{', '}', '[', ']',
  '?', 'r', 'c', 'v', 'z', 'x', 'Y', 'U', 'J', 'C', 'L', 'Q', '0', 'O', 'Z', 'm', 'w', 'q',
  'p', 'd', 'b', 'k', 'h', 'a', 'o', '*', '#', 'M', 'W', '&', '8', '%', '@',
];

function reduced() {
  return matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export function mountAscii(host, getSource, opts = {}) {
  // Columns follow the panel's real width: a fixed count that reads as a transcription on a
  // desktop hero is 3px-per-glyph mud on a phone. ~13 CSS px per column keeps a glyph a glyph.
  const maxCols = opts.cols ?? 150;
  // Cell size is per treatment: a glyph has to be big enough to read as a character, a dither
  // cell has to be small enough to disappear into an image.
  const CELL_PX = { glyphs: 14, dither: 3.4, edges: 4.5, bloom: 2.6 };
  const modeNow = () => (typeof window !== 'undefined' && window.__heroMode) || opts.mode || 'glyphs';
  const colsFor = (w) => Math.max(44, Math.min(1400, Math.round(w / (opts.cellPx ?? CELL_PX[modeNow()] ?? 14))));
  let cols = colsFor(host.clientWidth || 1200);
  const fps = reduced() ? 4 : opts.fps ?? 15;

  const layer = document.createElement('canvas');
  layer.className = 'ascii-layer';
  layer.setAttribute('aria-hidden', 'true'); // decorative: the frame underneath is the content
  // sits directly over the film, under everything the page draws on top of it (shade, copy, HUD)
  if (opts.after?.parentNode === host) host.insertBefore(layer, opts.after.nextSibling);
  else host.appendChild(layer);

  // one small offscreen buffer, reused: the sample grid is the character grid, so the browser's
  // own downscale does the averaging and no per-frame allocation happens in the loop.
  const scratchEl = document.createElement('canvas'); // 1:1 bit buffer for the dither treatment
  const scratch = { el: scratchEl, ctx: scratchEl.getContext('2d') };
  const hotEl = document.createElement('canvas'); // the bright cells only, for the bloom pass
  const hotBuf = { el: hotEl, ctx: hotEl.getContext('2d') };
  const sample = document.createElement('canvas');
  const sctx = sample.getContext('2d', { willReadFrequently: true });
  const lctx = layer.getContext('2d');

  let rows = 0;
  let rowsPerCell = 1.55;
  let cell = 0;
  let dpr = 1;
  let raf = 0;
  let last = 0;
  let stopped = false;

  function size(src) {
    const w = host.clientWidth;
    const h = host.clientHeight;
    if (!w || !h || !src?.width) return false;
    dpr = Math.min(2, devicePixelRatio || 1);
    cols = colsFor(w);
    rowsPerCell = modeNow() === 'glyphs' ? 1.55 : 1; // square cells for the pixel treatments
    cell = w / cols;
    rows = Math.max(1, Math.round(h / (cell * rowsPerCell))); // glyph cells are ~1.55x taller than wide
    if (layer.width !== Math.round(w * dpr) || layer.height !== Math.round(h * dpr)) {
      layer.width = Math.round(w * dpr);
      layer.height = Math.round(h * dpr);
      layer.style.width = `${w}px`;
      layer.style.height = `${h}px`;
    }
    if (sample.width !== cols || sample.height !== rows) {
      sample.width = cols;
      sample.height = rows;
    }
    return true;
  }

  // Per-frame auto-levels. A fixed tone curve was the reason the field read as texture and not as
  // a scene: DOOM's palette is dark and narrow, so a corridor, a wall and a doorway all landed in
  // the same two or three glyphs. Stretching each frame's own 2nd–98th percentile across the full
  // ramp gives the scene back its structure, and smoothing the bounds across frames stops the
  // picture pumping when a fireball lights the room.
  const HIST = new Uint32Array(64);
  let loSm = 0;
  let hiSm = 1;
  function levels(px, n) {
    HIST.fill(0);
    for (let i = 0; i < n; i += 4) {
      const l = (px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114) / 255;
      HIST[Math.min(63, (l * 64) | 0)] += 1;
    }
    const total = n / 4;
    const want = total * 0.02;
    let acc = 0;
    let lo = 0;
    let hi = 63;
    for (let b = 0; b < 64; b++) { acc += HIST[b]; if (acc >= want) { lo = b; break; } }
    acc = 0;
    for (let b = 63; b >= 0; b--) { acc += HIST[b]; if (acc >= want) { hi = b; break; } }
    const loV = lo / 64;
    const hiV = Math.max(loV + 0.08, (hi + 1) / 64);
    loSm += (loV - loSm) * 0.25; // ~4-frame smoothing
    hiSm += (hiV - hiSm) * 0.25;
  }

  function draw(src) {
    sctx.drawImage(src, 0, 0, cols, rows);
    const px = sctx.getImageData(0, 0, cols, rows).data;
    levels(px, px.length);
    const span = Math.max(0.05, hiSm - loSm);
    const mode = modeNow();
    if (mode !== 'glyphs') {
      lctx.setTransform(1, 0, 0, 1, 0, 0);
      lctx.clearRect(0, 0, layer.width, layer.height);
      if (mode === 'edges') paintEdges(lctx, px, cols, rows, layer.width, layer.height, loSm, span);
      else if (mode === 'bloom') paintBloom(lctx, px, cols, rows, layer.width, layer.height, loSm, span, hotBuf);
      else paintDither(lctx, px, cols, rows, layer.width, layer.height, loSm, span, scratch, hotBuf);
      return;
    }
    const W = layer.width;
    const H = layer.height;
    lctx.setTransform(1, 0, 0, 1, 0, 0);
    lctx.clearRect(0, 0, W, H);
    // no opaque backing: the frame stays faintly visible through its own transcription, which is
    // both better looking and closer to the point (the picture is there; the model isn't reading it)
    // A heavy backing hid the scene: the glyphs alone cannot show you a zombieman four cells
    // ahead. The transcription is a veil over the frame, not a replacement for it — the picture
    // stays legible and the characters sit on top of it.
    lctx.fillStyle = 'rgba(8,8,9,0.22)';
    lctx.fillRect(0, 0, W, H);
    const cw = W / cols;
    const ch = H / rows;
    lctx.font = `${(ch * 1.02).toFixed(1)}px "Geist Mono", ui-monospace, monospace`;
    lctx.textBaseline = 'middle';
    lctx.textAlign = 'center';
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const i = (y * cols + x) * 4;
        // Rec. 601 luma, then a gamma lift: DOOM's palette sits dark and a linear ramp maps most
        // of a corridor onto the same two glyphs.
        const r = px[i];
        const gch = px[i + 1];
        const bch = px[i + 2];
        const raw = (r * 0.299 + gch * 0.587 + bch * 0.114) / 255;
        // DOOM's palette sits dark and flat: a linear ramp maps a whole corridor onto two glyphs
        // and the field reads as static. Stretch the band the frame actually occupies.
        // A 2x2 ordered dither before the glyph lookup: without it, a large flat wall quantises to
        // one character and the field draws in horizontal bands of repeated letters.
        const d = (BAYER[(y & 1) * 2 + (x & 1)] - 0.5) / RAMP.length;
        const l = Math.max(0, Math.min(1, (raw - loSm) / span)) ** 0.95 + d;
        // Everything below the floor draws nothing. This is what makes the field legible as a
        // scene rather than a wall of characters: unlit geometry stays empty, so the shapes that
        // are lit — a doorway, a lamp, a wall the player is facing — are the only things written.
        if (l < FLOOR) continue;
        const t = (l - FLOOR) / (1 - FLOOR);
        const g = RAMP[Math.max(0, Math.min(RAMP.length - 1, Math.floor(t * RAMP.length)))];
        if (g === ' ') continue;
        // keep the frame's own colour, lifted: a monochrome field loses the one thing the picture
        // still carries at this resolution (a red wall, a green lamp, brown brick)
        const lift = raw > 0.01 ? Math.min(2.8, (0.45 + 0.8 * t) / raw) : 1;
        lctx.fillStyle = `rgb(${Math.min(255, r * lift) | 0},${Math.min(255, gch * lift) | 0},${Math.min(255, bch * lift) | 0})`;
        lctx.globalAlpha = 0.3 + 0.5 * t;
        lctx.fillText(g, (x + 0.5) * cw, (y + 0.5) * ch);
      }
    }
    lctx.globalAlpha = 1;
  }

  function frame(t) {
    if (stopped) return;
    raf = requestAnimationFrame(frame);
    if (t - last < 1000 / fps) return;
    last = t;
    const src = getSource();
    if (!src || !size(src)) return;
    try {
      draw(src);
    } catch {
      // a source that can't be read (cross-origin, no preserved buffer) gets no overlay at all
      // rather than a black box over the film
      stop();
      layer.remove();
    }
  }

  function stop() {
    stopped = true;
    cancelAnimationFrame(raf);
  }

  raf = requestAnimationFrame(frame);

  // No reveal gesture at all. A whole-stage hover was wrong (the cursor rests over the hero while
  // you read, so the layer was permanently dissolved), and a cursor lens was a gimmick on top of
  // it. The transcription is the hero: it stays up, full bleed, and the copy gets its own scrim
  // rather than a hole cut in the picture.
  return { stop };
}

// ---------------------------------------------------------------------------------------------
// Alternative hero treatments, for picking between. Each takes the sampled frame and paints the
// layer; they share the sampler, the resize handling and the mount above.
//
//   'dither' — one bit per cell, 4x4 ordered. The frame reduced to the least information that
//              still shows you the room: it reads instantly, and "almost nothing left" is the
//              point the page is making.
//   'edges'  — Sobel gradient, drawn as light on black. Machine vision: structure only, no
//              surfaces. Legible at a glance and unmistakably not a photograph.
// ---------------------------------------------------------------------------------------------

// Bloom only: no quantisation at all. The layer holds nothing but the halation from the bright
// parts of the frame, composited additively over the sharp film underneath — what a camera does
// to a lit corridor, not a filter over it.
export function paintBloom(lctx, px, cols, rows, W, H, lo, span, hot) {
  const glow = hot.ctx.createImageData(cols, rows);
  const g = glow.data;
  for (let i = 0; i < px.length; i += 4) {
    const raw = (px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114) / 255;
    const l = Math.max(0, Math.min(1, (raw - lo) / span));
    const a = l > HOT ? Math.min(1, (l - HOT) / (1 - HOT)) : 0;
    // keep the source colour in the glow so a green lamp blooms green, not white
    g[i] = Math.min(255, px[i] * 1.15);
    g[i + 1] = Math.min(255, px[i + 1] * 1.15);
    g[i + 2] = Math.min(255, px[i + 2] * 1.15);
    g[i + 3] = Math.round(255 * a);
  }
  if (hot.el.width !== cols || hot.el.height !== rows) {
    hot.el.width = cols;
    hot.el.height = rows;
  }
  hot.ctx.putImageData(glow, 0, 0);
  lctx.save();
  lctx.globalCompositeOperation = 'lighter';
  lctx.imageSmoothingEnabled = true;
  lctx.globalAlpha = 0.55;
  lctx.filter = 'blur(10px)';
  lctx.drawImage(hot.el, 0, 0, W, H);
  lctx.globalAlpha = 0.32;
  lctx.filter = 'blur(34px)';
  lctx.drawImage(hot.el, 0, 0, W, H);
  lctx.restore();
  lctx.filter = 'none';
}

export function paintDither(lctx, px, cols, rows, W, H, lo, span, buf, hot) {
  // One fillRect per cell would be ~100k calls a frame at this resolution. Write the bits into an
  // ImageData instead and let the compositor scale it up with smoothing off: same picture, one
  // draw call, and the cell edges stay hard.
  const img = buf.ctx.createImageData(cols, rows);
  const glow = hot.ctx.createImageData(cols, rows);
  const d = img.data;
  const g = glow.data;
  for (let y = 0, n = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++, n++) {
      const i = n * 4;
      const raw = (px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114) / 255;
      const l = Math.max(0, Math.min(1, (raw - lo) / span));
      const on = l > BAYER4[(y & 3) * 4 + (x & 3)];
      d[i] = d[i + 1] = on ? 240 : 11;
      d[i + 2] = on ? 235 : 12;
      d[i + 3] = 255;
      // Only the genuinely bright cells bloom. Blooming every lit cell — and at this resolution
      // half of them are lit — just raises the black level and greys the whole frame out.
      const h = on && l > HOT ? Math.round(255 * Math.min(1, (l - HOT) / (1 - HOT))) : 0;
      g[i] = g[i + 1] = g[i + 2] = h;
      g[i + 3] = h;
    }
  }
  for (const b of [buf, hot]) {
    if (b.el.width !== cols || b.el.height !== rows) {
      b.el.width = cols;
      b.el.height = rows;
    }
  }
  buf.ctx.putImageData(img, 0, 0);
  hot.ctx.putImageData(glow, 0, 0);
  lctx.imageSmoothingEnabled = false;
  lctx.drawImage(buf.el, 0, 0, W, H);
  lctx.save();
  lctx.imageSmoothingEnabled = true;
  lctx.globalCompositeOperation = 'lighter';
  lctx.globalAlpha = 0.55;
  lctx.filter = 'blur(6px)';
  lctx.drawImage(hot.el, 0, 0, W, H);
  lctx.globalAlpha = 0.3;
  lctx.filter = 'blur(22px)';
  lctx.drawImage(hot.el, 0, 0, W, H);
  lctx.restore();
  lctx.filter = 'none';
}

export function paintEdges(lctx, px, cols, rows, W, H, lo, span) {
  const cw = W / cols;
  const ch = H / rows;
  lctx.fillStyle = '#0b0b0c';
  lctx.fillRect(0, 0, W, H);
  const lum = new Float32Array(cols * rows);
  for (let i = 0, n = 0; i < px.length; i += 4, n++) {
    const raw = (px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114) / 255;
    lum[n] = Math.max(0, Math.min(1, (raw - lo) / span));
  }
  const at = (x, y) => lum[Math.min(rows - 1, Math.max(0, y)) * cols + Math.min(cols - 1, Math.max(0, x))];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      // Sobel, both axes
      const gx = at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1);
      const gy = at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1);
      const m = Math.min(1, Math.hypot(gx, gy) * 0.9);
      if (m < 0.12) continue;
      lctx.globalAlpha = 0.25 + 0.75 * m;
      lctx.fillStyle = '#f0eeeb';
      lctx.fillRect(x * cw, y * ch, Math.max(1, cw * 0.9), Math.max(1, ch * 0.9));
    }
  }
  lctx.globalAlpha = 1;
}
