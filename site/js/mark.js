// mark.js — the hero tile mosaic. 12x12 grid of champagne squares; each tile's opacity is a
// real probability pulled from data/replays.json (first 144 probability values, walked in
// file/object order — not sorted into a gradient). The mark IS a decision field: every
// square is a number the model actually produced, not a decorative pattern.
const GRID = 12;
const N = GRID * GRID;

async function loadProbs() {
  try {
    const res = await fetch('data/replays.json');
    if (!res.ok) return [];
    const replays = await res.json();
    const vals = [];
    outer: for (const entry of Object.values(replays)) {
      for (const r of entry.results || []) {
        for (const p of Object.values(r.probs || {})) {
          vals.push(p);
          if (vals.length >= N) break outer;
        }
      }
    }
    return vals;
  } catch {
    return [];
  }
}

export async function mount(el) {
  if (!el) return;
  el.classList.add('mark');
  const probs = await loadProbs();
  el.innerHTML = '';
  for (let i = 0; i < N; i++) {
    const tile = document.createElement('span');
    tile.className = 'mark-tile';
    // ponytail: fallback opacity if replays.json is missing/short — still reads as a field,
    // just not a real one. Floor at .06 so no tile disappears entirely.
    const p = probs[i] ?? 0.15;
    tile.style.opacity = Math.max(0.1, Math.min(1, p)).toFixed(3);
    el.appendChild(tile);
  }
}
