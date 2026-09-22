// motion.js — the choreography, and deliberately almost nothing. GSAP 3 + ScrollTrigger, vendored
// in js/vendor/. The hero enters once per session; on scroll, only chapter heads rise. Nothing
// animates layout properties. Reduced motion: everything renders in its final state.

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
  // Only the chapter heads move. The earlier pass faded up every media panel, card, demo row and
  // destination link and counted every figure from zero — the stock scroll-reveal kit, and the
  // count-up turned .804 into a slot machine. A figure a visitor is meant to trust should be
  // printed, not animated.
}

export function mountMotion() {
  if (!gs() || reduced()) return;
  heroEnter();
  scrollReveals();
}
