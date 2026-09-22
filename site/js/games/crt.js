// Shared phosphor-terminal helpers for game canvases: scanlines, a cheap glyph bloom and the
// persistence-decay curve. Pure canvas-2d functions operating on a caller-owned ctx - no DOM,
// no per-instance state, so a draw loop can call these every frame without allocating. Only
// render-snake.js uses this today; kept generic (not Snake-specific) in case Drive/Doom want
// the same CRT treatment later.

// Darkens alternate device-pixel rows across [x,y,w,h] - the scanline artifact, drawn as flat
// fills (not a gradient) so it reads as a screen property rather than page chrome.
export function drawScanlines(ctx, x, y, w, h, { step = 3, alpha = 0.14 } = {}) {
  ctx.save();
  ctx.fillStyle = `rgba(0,0,0,${alpha})`;
  ctx.beginPath();
  for (let sy = y; sy < y + h; sy += step) {
    ctx.rect(x, sy, w, Math.max(1, step / 2));
  }
  ctx.fill();
  ctx.restore();
}

// Draws `ch` twice: an oversized, dim copy first (the glow fringe), then the crisp glyph on
// top - cheaper than shadowBlur when called for every lit cell on the board, and looks the
// same at this scale. `font`/`haloFont` are pre-built strings (cache them on resize, not per
// frame) so this never touches ctx.font more than the two swaps it needs.
export function drawGlyphBloom(ctx, ch, cx, cy, { font, haloFont, color, haloColor }) {
  ctx.font = haloFont;
  ctx.fillStyle = haloColor;
  ctx.fillText(ch, cx, cy);
  ctx.font = font;
  ctx.fillStyle = color;
  ctx.fillText(ch, cx, cy);
}

// Phosphor decay: 1 the instant a cell goes dark, 0 once `ms` (~2 ticks) has passed. Callers
// multiply this into the fading glyph's alpha.
export function phosphorDecay(elapsedMs, ms) {
  return Math.max(0, 1 - elapsedMs / ms);
}

export function prefersReducedMotion() {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function selfTest() {
  console.assert(phosphorDecay(0, 400) === 1, 'phosphorDecay starts fully lit');
  console.assert(phosphorDecay(400, 400) === 0, 'phosphorDecay reaches 0 at ms');
  console.assert(phosphorDecay(800, 400) === 0, 'phosphorDecay clamps past ms');
  console.assert(phosphorDecay(200, 400) === 0.5, 'phosphorDecay is linear');
  console.log('crt.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };

// ---------------------------------------------------------------------------------------------
// The tube itself. Without these the phosphor grid is just green text on black; a CRT reads as a
// CRT because of the glass around it — a bezel, a vignette into the corners, the slow refresh
// band, and the colour fringe where the three guns fail to converge.
// ---------------------------------------------------------------------------------------------

// Rounded bezel + inner vignette around a screen rect. Draw before the glyphs (the vignette is
// drawn again after, as `drawGlass`, so the corners darken the content too).
export function drawBezel(ctx, x, y, w, h, { radius = 14, bezel = 10 } = {}) {
  ctx.save();
  ctx.fillStyle = '#17181a';
  roundRect(ctx, x - bezel, y - bezel, w + bezel * 2, h + bezel * 2, radius + 4);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = '#05070600';
  ctx.restore();
}

// Vignette + refresh band + a faint scan glow, drawn over the glyphs. `t` is a timestamp in ms.
export function drawGlass(ctx, x, y, w, h, { t = 0, radius = 14, roll = true } = {}) {
  ctx.save();
  roundRect(ctx, x, y, w, h, radius);
  ctx.clip();
  // corner falloff: a radial darkening, the curvature cue that costs nothing
  const g = ctx.createRadialGradient(x + w / 2, y + h / 2, Math.min(w, h) * 0.22, x + w / 2, y + h / 2, Math.max(w, h) * 0.72);
  g.addColorStop(0, 'rgba(0,0,0,0)');
  g.addColorStop(1, 'rgba(0,0,0,0.55)');
  ctx.fillStyle = g;
  ctx.fillRect(x, y, w, h);
  if (roll) {
    // refresh band: one soft bright line sweeping down about every 7 s
    const bandY = y + ((t / 7000) % 1) * (h + 120) - 60;
    const band = ctx.createLinearGradient(0, bandY - 60, 0, bandY + 60);
    band.addColorStop(0, 'rgba(180,255,210,0)');
    band.addColorStop(0.5, 'rgba(180,255,210,0.045)');
    band.addColorStop(1, 'rgba(180,255,210,0)');
    ctx.fillStyle = band;
    ctx.fillRect(x, bandY - 60, w, 120);
  }
  ctx.restore();
}

// Misconvergence: the same glyph smeared a fraction of a pixel left in red and right in blue.
// Call instead of a plain fillText for the brightest cells only — it is three draws per glyph.
export function drawFringe(ctx, ch, cx, cy, { spread = 0.8, alpha = 0.22 } = {}) {
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.fillStyle = `rgba(255,60,60,${alpha})`;
  ctx.fillText(ch, cx - spread, cy);
  ctx.fillStyle = `rgba(60,120,255,${alpha})`;
  ctx.fillText(ch, cx + spread, cy);
  ctx.restore();
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
