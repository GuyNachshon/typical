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
  if (reduced() || sessionStorage.getItem('typical_hero')) return;
  sessionStorage.setItem('typical_hero', '1');
  // The HUD used to arrive on opacity alone -- it materialised out of nothing, which is the one
  // entrance nothing in the world makes. Both cards now come from slightly below and slightly
  // small, so they read as arriving rather than appearing. Transform and opacity only: both are
  // composited, so a 60fps entrance survives the wasm build booting underneath it.
  g.set([...parts, hud].filter(Boolean), { opacity: 0 });
  g.set(parts, { y: 12 });
  g.set(hud, { y: 14, scale: 0.985, transformOrigin: '100% 100%' }); // anchored to its own corner
  const tl = g.timeline({ defaults: { ease: 'expo.out' } });
  // 50ms between lines: enough to read as a cascade, short enough not to feel like a queue
  tl.to(parts, { opacity: 1, y: 0, duration: 0.52, stagger: 0.05 }, 0.15)
    .to(hud, { opacity: 1, y: 0, scale: 1, duration: 0.46 }, 0.42);
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

  // These are created after the cold start, which can be many seconds in — and the reader has been
  // scrolling the whole time. gsap.from() writes its start state the instant it is called, so
  // applying it to something already on screen blanks it, and the `once` trigger can never play it
  // back because the element is already past 'top 82%'. Anything at or above that line is therefore
  // left exactly as it is: already visible is already correct. This is what makes the choreography
  // safe to mount at any moment rather than only at the one it was designed for.
  const pending = (el) => el.getBoundingClientRect().top > innerHeight * 0.82;

  document.querySelectorAll('.chapter-head').forEach((head) => {
    if (!pending(head)) return;
    const items = head.querySelectorAll('.t-eyebrow, .t-section, .lede, .t-hero');
    g.from(items, { opacity: 0, y: 14, duration: 0.6, ease: 'expo.out', stagger: 0.07, scrollTrigger: { trigger: head, ...once } });
  });
  // The post has its own structure -- .chapter > .claim, not .chapter-head -- so none of the above
  // ever matched it and the long-form page had no choreography at all. One reveal per chapter:
  // the rule draws (CSS, above), then the heading and its standfirst rise behind it. The figures
  // are deliberately left alone. A chart a reader is meant to trust should be printed, not
  // performed, and the same argument that keeps the count-up off the numbers keeps the fade off
  // the figures.
  document.querySelectorAll('.post .chapter').forEach((chapter) => {
    if (!pending(chapter)) return;
    const items = chapter.querySelectorAll(':scope > .claim > h2, :scope > .claim > p');
    if (!items.length) return;
    chapter.classList.add('will-reveal');
    g.from(items, {
      opacity: 0,
      y: 12,
      duration: 0.55,
      ease: 'expo.out',
      stagger: 0.06,
      scrollTrigger: {
        trigger: chapter,
        ...once,
        onEnter: () => chapter.classList.add('is-revealed'),
      },
    });
  });

  // Only the chapter heads move. The earlier pass faded up every media panel, card, demo row and
  // destination link and counted every figure from zero — the stock scroll-reveal kit, and the
  // count-up turned .804 into a slot machine. A figure a visitor is meant to trust should be
  // printed, not animated.

  // Results bars: "the gap between the rule and the tick is the result", so each fill sweeps from
  // its own baseline tick, not from zero — scaleX from a per-element ratio, never width (see the
  // header comment on layout properties). The numerals are never touched, same reasoning as above.
  [
    ['#results-table', '.score-track'],
    ['#results-compare', '.rank-track'],
  ].forEach(([panelSel, trackSel]) => {
    const panel = document.querySelector(panelSel);
    if (!panel || !pending(panel)) return;
    const fills = [...panel.querySelectorAll(`${trackSel} i`)].filter((el) => {
      const tick = el.nextElementSibling;
      if (!tick || tick.tagName !== 'B') return false;
      const ratio = parseFloat(tick.style.left) / parseFloat(el.style.width);
      return Number.isFinite(ratio) && ratio <= 1;
    });
    if (!fills.length) return;
    g.from(fills, {
      scaleX: (i, el) => parseFloat(el.nextElementSibling.style.left) / parseFloat(el.style.width),
      transformOrigin: 'left center',
      duration: 0.7,
      ease: 'power2.out',
      stagger: 0.06,
      scrollTrigger: { trigger: panel, ...once },
    });
  });
}

// gsap.from() writes its start state the moment it is called, so calling this twice re-hides every
// chapter head — and the second set of `once: true` triggers never fires for anything already
// scrolled past, leaving the text at opacity 0 for good. The boot path had two racing callers and
// produced exactly that, twelve seconds in. One guard here covers every caller, present and future.
let mounted = false;

export function mountMotion() {
  if (mounted || !gs() || reduced()) return;
  mounted = true;
  heroEnter();
  heroInset();
  scrollReveals();
}
