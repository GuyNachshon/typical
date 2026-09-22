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
  const colsFor = (w) => Math.max(44, Math.min(maxCols, Math.round(w / (opts.cellPx ?? 9))));
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
  const sample = document.createElement('canvas');
  const sctx = sample.getContext('2d', { willReadFrequently: true });
  const lctx = layer.getContext('2d');

  let rows = 0;
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
    cell = w / cols;
    rows = Math.max(1, Math.round(h / (cell * 1.55))); // glyph cells are ~1.55x taller than wide
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

  function draw(src) {
    sctx.drawImage(src, 0, 0, cols, rows);
    const px = sctx.getImageData(0, 0, cols, rows).data;
    const W = layer.width;
    const H = layer.height;
    lctx.setTransform(1, 0, 0, 1, 0, 0);
    lctx.clearRect(0, 0, W, H);
    // no opaque backing: the frame stays faintly visible through its own transcription, which is
    // both better looking and closer to the point (the picture is there; the model isn't reading it)
    lctx.fillStyle = 'rgba(8,8,9,0.93)';
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
        const l = Math.max(0, Math.min(1, (raw - 0.05) / 0.62)) ** 1.1 + d;
        const g = RAMP[Math.max(0, Math.min(RAMP.length - 1, Math.floor(l * RAMP.length)))];
        if (g === ' ') continue;
        // keep the frame's own colour, lifted: a monochrome field loses the one thing the picture
        // still carries at this resolution (a red wall, a green lamp, brown brick)
        const lift = raw > 0.01 ? Math.min(2.4, 0.55 / raw + 0.55) : 1;
        lctx.fillStyle = `rgb(${Math.min(255, r * lift) | 0},${Math.min(255, gch * lift) | 0},${Math.min(255, bch * lift) | 0})`;
        lctx.globalAlpha = 0.45 + 0.55 * l;
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
