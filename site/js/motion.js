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

// The hero insets as you leave it: the full-bleed film scales down a little and its corners
// round, so the page reads as scrolling past a card rather than wiping a video off the top.
// Scrubbed, not triggered — the shape tracks the scroll position exactly. Transform and
// border-radius only; the stage keeps its layout box, so nothing below it moves.
function heroInset() {
  const g = gs();
  const ST = window.ScrollTrigger;
  const stage = document.querySelector('.stage');
  if (!ST || !stage) return;
  g.registerPlugin(ST);
  // Pinned: the first stretch of scroll is spent insetting the hero in place, and only once the
  // card has settled does the page start moving underneath it. ScrollTrigger inserts a spacer of
  // the pinned distance, so nothing below overlaps.
  g.set(stage, { transformOrigin: '50% 50%' });
  g.to(stage, {
    scale: 0.93,
    borderRadius: 18,
    ease: 'none',
    scrollTrigger: {
      // the track is 155vh and the stage is sticky inside it, so the hero holds on its own; this
      // only drives the shape. No pin: pinning re-parents the element, and that reloads the film.
      trigger: stage.parentElement || stage,
      start: 'top top',
      end: '+=55%',
      scrub: 0.35,
      invalidateOnRefresh: true,
    },
  });
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
  heroInset();
  scrollReveals();
}
