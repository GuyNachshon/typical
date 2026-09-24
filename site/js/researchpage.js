// researchpage.js -- mounts for the research page.
//
// The page is the paper with the notation taken out, so the figures are the paper's figures redrawn
// on paper stock rather than for print. Data comes from site/data/paper-*.json, transcribed from the
// paper on branch main; nothing here recomputes a result.

import { mountFlow } from './arch.js';
import { mountDeltaQ, mountReadouts, mountLadder, mountDepth, mountRender } from './paperfigs.js';
import { costBar } from './charts.js';

const load = (p) => fetch(p).then((r) => (r.ok ? r.json() : null)).catch(() => null);

// Figure numbers from document order, the same rule the launch post uses: a caption that names its
// own number goes stale the first time a section moves.
function numberFigures() {
  [...document.querySelectorAll('.post figure.fig')].forEach((fig, i) => {
    const cap = fig.querySelector('figcaption');
    if (!cap || cap.dataset.numbered) return;
    cap.dataset.numbered = '1';
    const b = document.createElement('b');
    b.textContent = `Fig. ${i + 1}.`;
    cap.prepend(b, ' ');
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  numberFigures();
  mountFlow(document.getElementById('chart-arch'));

  const [dq, ro, lad, lat, dep, ren] = await Promise.all([
    load('data/paper-deltaq.json'),
    load('data/paper-readouts.json'),
    load('data/paper-ladder.json'),
    load('data/latency.json'),
    load('data/paper-depth.json'),
    load('data/paper-render.json'),
  ]);

  if (ro) mountReadouts(document.getElementById('chart-readouts'), ro);
  if (lad) mountLadder(document.getElementById('chart-ladder'), lad, 'std');
  if (dep) mountDepth(document.getElementById('chart-depth'), dep);
  if (ren) mountRender(document.getElementById('chart-render'), ren);

  if (dq) {
    const host = document.getElementById('chart-deltaq');
    const fig = mountDeltaQ(host, dq);
    const btn = document.getElementById('dq-reveal');
    // The reveal is the argument: the flat view is what a leaderboard shows you, and the control is
    // what it does not. It runs once on its own when the figure is reached, because a reader who
    // never presses the button should still see the point -- and the button then puts it back.
    let open = false;
    const flip = () => {
      open = !open;
      fig.open();
      btn.textContent = open ? 'Accuracy only' : 'Add the control';
    };
    btn?.addEventListener('click', flip);
    let timer = 0;
    btn?.addEventListener('click', () => clearTimeout(timer), { once: true });
    const io = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting) && !open) {
        io.disconnect();
        timer = setTimeout(() => { if (!open) flip(); }, 900);
      }
    }, { threshold: 0.45 });
    io.observe(host);
  }

  // costBar draws exactly this shape: what the first decision costs and what the next 31 add
  const rows = Object.entries(lat?.ladder || {})
    .filter(([, l]) => l?.single_ms?.k2 != null && l?.marginal_ms_m32?.k2 != null)
    .map(([model, l]) => ({ model, one: l.single_ms.k2, marginal: l.marginal_ms_m32.k2 }));
  if (rows.length) {
    costBar(document.getElementById('chart-latency'), {
      rows, m: 32, title: 'One decision on a cold state, and each question after it on the same cache',
    });
  }
});
