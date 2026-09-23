// release.js — the launch post. Reuses shell.js (loaded as its own <script type="module"> tag,
// unchanged, right before this one) for everything it already does: the hero and card DOOM films,
// the drive/snake game screens, and the cost chart on #chart-g2 via mountResults -> costBar off
// site/data/models.json. This page reuses those same element ids on purpose so shell.js's boot()
// mounts them with no page-specific code here.
//
// The one chart shell.js doesn't own is the read-path diagram under "One state, many decisions" —
// mount it directly.
import { mountFlow } from './arch.js';

document.addEventListener('DOMContentLoaded', () => {
  const el = document.getElementById('chart-arch');
  if (el) mountFlow(el);
});
