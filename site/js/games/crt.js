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
