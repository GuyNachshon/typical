// Thermal palette: cold navy -> magenta -> orange -> white, used for every
// probability/heat encoding on the site (charts, snake bars, demo output).
const STOPS = ['#0b1035', '#3b1a7a', '#8a1e8a', '#d63a5c', '#ff7a1a', '#ffd23f', '#fff7d6'];

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

// thermal(p) -> css rgb() color for p in [0,1], interpolated through STOPS.
export function thermal(p) {
  p = clamp01(p);
  const idx = p * (STOPS.length - 1);
  const i0 = Math.floor(idx);
  const i1 = Math.min(STOPS.length - 1, i0 + 1);
  const t = idx - i0;
  const c0 = hexToRgb(STOPS[i0]);
  const c1 = hexToRgb(STOPS[i1]);
  const r = Math.round(c0.r + (c1.r - c0.r) * t);
  const g = Math.round(c0.g + (c1.g - c0.g) * t);
  const b = Math.round(c0.b + (c1.b - c0.b) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

const SHADES = ['░', '▒', '▓', '█'];

// thermalBlocks(p, n) -> text-mode bar of n shade glyphs, filled left-to-right
// proportional to p. ponytail: one glyph family (shade blocks) covers both the
// "░▒▓█" and "▁▂▃▅▇█" asks in the brief; add height variant if a spot needs it.
export function thermalBlocks(p, n = 12) {
  p = clamp01(p);
  const filled = p * n;
  let out = '';
  for (let i = 0; i < n; i++) {
    const level = clamp01(filled - i);
    out += level <= 0 ? '░' : SHADES[Math.min(3, Math.floor(level * 4))];
  }
  return out;
}

function selfTest() {
  console.assert(thermal(0) === 'rgb(11, 16, 53)', 'thermal(0) is the first stop');
  console.assert(thermal(1) === 'rgb(255, 247, 214)', 'thermal(1) is the last stop');
  console.assert(thermalBlocks(0, 4) === '░░░░', 'thermalBlocks(0) is all empty');
  console.assert(thermalBlocks(1, 4) === '████', 'thermalBlocks(1) is all full');
  console.assert(thermalBlocks(0.5, 4).length === 4, 'thermalBlocks respects n');
  console.log('thermal.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
