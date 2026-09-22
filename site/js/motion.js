// motion.js — the choreography. GSAP 3 + ScrollTrigger from CDN (loaded in index.html).
// Intent, not decoration: the hero enters once per session; on scroll, chapter heads rise, media
// panels settle, figures count up, cards stagger. Nothing animates layout properties. Reduced
// motion: everything renders in its final state.

const gs = () => window.gsap;

function reduced() { return matchMedia('(prefers-reduced-motion: reduce)').matches; }

function heroEnter() {
  const g = gs();
  const copy = document.querySelector('.stage-copy');
  if (!copy) return;
  const parts = [...copy.children];
  const hud = document.querySelector('.stage-hud');
  const nav = document.querySelector('.nav');
  if (reduced() || sessionStorage.getItem('typical_hero')) return;
  sessionStorage.setItem('typical_hero', '1');
  g.set([...parts, hud, nav].filter(Boolean), { opacity: 0 });
  g.set(parts, { y: 18 });
  const tl = g.timeline({ defaults: { ease: 'expo.out' } });
  tl.to(nav, { opacity: 1, duration: 0.4 }, 0)
    .to(parts, { opacity: 1, y: 0, duration: 0.7, stagger: 0.08 }, 0.15)
    .to(hud, { opacity: 1, duration: 0.5 }, 0.7);
}

function countUp(el) {
  const g = gs();
  const raw = el.textContent.trim();
  const m = raw.match(/^(\.?\d+(?:\.\d+)?)/);
  if (!m) return;
  const target = parseFloat(m[1]);
  const decimals = (m[1].split('.')[1] || '').length;
  const leadingDot = m[1].startsWith('.');
  const rest = raw.slice(m[1].length);
  const suffixNode = el.querySelector('span');
  const obj = { v: 0 };
  g.to(obj, {
    v: target, duration: 1.1, ease: 'expo.out',
    onUpdate() {
      let s = obj.v.toFixed(decimals);
      if (leadingDot) s = s.replace(/^0/, '');
      el.firstChild.nodeValue = s;
    },
    onComplete() { el.firstChild.nodeValue = leadingDot ? m[1] : m[1]; },
  });
  void rest; void suffixNode;
}

function scrollReveals() {
  const g = gs();
  const ST = window.ScrollTrigger;
  if (!ST) return;
  g.registerPlugin(ST);
  const once = { once: true, start: 'top 82%' };

  document.querySelectorAll('.chapter-head').forEach((head) => {
    const items = head.querySelectorAll('.t-eyebrow, .t-section, .lede, .t-hero');
    g.from(items, { opacity: 0, y: 14, duration: 0.6, ease: 'expo.out', stagger: 0.07, scrollTrigger: { trigger: head, ...once } });
  });
  document.querySelectorAll('.media').forEach((m) => {
    g.from(m, { opacity: 0, scale: 0.985, duration: 0.7, ease: 'expo.out', transformOrigin: '50% 60%', scrollTrigger: { trigger: m, ...once } });
  });
  document.querySelectorAll('.ucard-grid').forEach((grid) => {
    g.from(grid.children, { opacity: 0, y: 16, duration: 0.6, ease: 'expo.out', stagger: 0.08, scrollTrigger: { trigger: grid, ...once } });
  });
  document.querySelectorAll('.t-figure').forEach((el) => {
    ST.create({ trigger: el, ...once, onEnter: () => countUp(el) });
  });
  document.querySelectorAll('.exrow').forEach((row, i) => {
    g.from(row, { opacity: 0, y: 10, duration: 0.5, ease: 'expo.out', delay: Math.min(i, 4) * 0.05, scrollTrigger: { trigger: row, ...once } });
  });
  document.querySelectorAll('.dest a').forEach((a) => {
    g.from(a, { opacity: 0, y: 12, duration: 0.6, ease: 'expo.out', scrollTrigger: { trigger: a, ...once } });
  });
}

export function mountMotion() {
  if (!gs() || reduced()) return;
  heroEnter();
  scrollReveals();
}
