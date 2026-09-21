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
    // label + argmax travel with each probability so a tile's tooltip can name the decision
    // it came from (there's no query text in replays.json, only the candidate/probability
    // pairs — the label is the honest, cheap thing to show).
    outer: for (const entry of Object.values(replays)) {
      for (const r of entry.results || []) {
        for (const [label, p] of Object.entries(r.probs || {})) {
          vals.push({ p, label, argmax: r.argmax });
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
    const entry = probs[i];
    const p = entry?.p ?? 0.15;
    // sqrt keeps the ordering but lets small probabilities read as texture instead of black
    tile.style.opacity = (0.08 + 0.92 * Math.sqrt(Math.max(0, Math.min(1, p)))).toFixed(3);
    if (entry) tile.title = `${entry.label} · p=${entry.p.toFixed(2)}${entry.label === entry.argmax ? ' (argmax)' : ''}`;
    el.appendChild(tile);
  }
}
