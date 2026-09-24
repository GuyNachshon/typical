// eqtoggle.js — an equation that can be asked to say itself in words.
//
// Each .eq holds two faces, the MathML and a plain-language sentence, and shows one at a time.
// Clicking swaps them; clicking again swaps back. The first pass at this was a hover popover
// floating above the equation, which meant the explanation was something that happened *to* the
// reader rather than something they asked for, and it covered the paragraph above while it was
// open. A toggle is the honest control: it stays where it is put.
//
//   mountEqToggles(root) -> wires every .eq under root
//
// The crossfade is masked with a short blur. Without it you see two sentences overlapping mid-swap
// and read neither; the blur bridges them so the eye takes it as one thing changing shape. Height
// is animated too, which breaks the usual "transform and opacity only" rule — there is no way to
// reflow a line of prose into a line of algebra without the box changing size, and this fires once
// per click rather than continuously, so it is the right place to spend it.

const DUR = 0.3;

const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

const LABEL = { math: 'click to simplify', plain: 'click for the maths' };

function wire(eq) {
  const faces = {
    math: eq.querySelector('[data-face="math"]'),
    plain: eq.querySelector('[data-face="plain"]'),
  };
  if (!faces.math || !faces.plain) return;
  const tip = eq.querySelector('.eq-tip');
  let showing = 'math';
  let busy = false;

  const sync = () => {
    if (tip) tip.textContent = LABEL[showing];
    eq.setAttribute('aria-expanded', String(showing === 'plain'));
    // The label is the accessible name of the control, so it has to say what the click will do.
    eq.setAttribute('aria-label', showing === 'math'
      ? 'Show this equation in plain language'
      : 'Show the equation');
  };
  sync();

  const toggle = () => {
    if (busy) return; // a second click mid-swap would measure a height that is still moving
    const next = showing === 'math' ? 'plain' : 'math';
    const from = faces[showing];
    const to = faces[next];
    const g = window.gsap;

    if (!g || reduced()) {
      from.hidden = true;
      to.hidden = false;
      showing = next;
      sync();
      return;
    }

    busy = true;
    const h0 = eq.offsetHeight;

    // The outgoing face leaves the flow before the incoming one is measured, so the box height is
    // read from the new content alone and nothing jumps. It stays painted, absolutely positioned
    // over its old spot, long enough to animate away.
    const style = getComputedStyle(eq);
    from.style.position = 'absolute';
    from.style.left = style.paddingLeft;
    from.style.right = style.paddingRight;
    from.style.top = style.paddingTop;
    from.style.pointerEvents = 'none';

    to.hidden = false;
    const h1 = eq.offsetHeight;

    const release = () => {
      from.hidden = true;
      from.style.position = from.style.left = from.style.right = from.style.top = from.style.pointerEvents = '';
      g.set(from, { clearProps: 'opacity,filter,y,clipPath' });
    };

    // Out is quick and up: the old line lifts, blurs and goes. Exits are faster than entrances,
    // because the reader has already decided and is waiting on the answer, not the departure.
    g.to(from, { opacity: 0, filter: 'blur(5px)', y: -8, duration: 0.2, ease: 'power2.out' });

    // In is a rewrite. The new face is wiped in left to right rather than faded up, which reads as
    // the line being written rather than swapped -- and it arrives slightly blurred so the two
    // never look like two separate things overlapping.
    g.fromTo(to,
      { opacity: 0, filter: 'blur(5px)', y: 10, clipPath: 'inset(0 100% 0 0)' },
      { opacity: 1, filter: 'blur(0px)', y: 0, clipPath: 'inset(0 0% 0 0)', duration: 0.46, ease: 'expo.out', delay: 0.06,
        // the gate reopens on the longest tween, not the shortest, or a fast second click would
        // land mid-rewrite and measure a height that is still moving
        onComplete: () => { busy = false; } });

    g.fromTo(eq, { height: h0 }, { height: h1, duration: 0.36, ease: 'power2.inOut', onComplete: () => { eq.style.height = ''; release(); } });

    showing = next;
    sync();
  };

  eq.addEventListener('click', toggle);
  eq.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
      e.preventDefault(); // space would otherwise scroll the page out from under the equation
      toggle();
    }
  });
}

export function mountEqToggles(root = document) {
  root.querySelectorAll('.eq').forEach(wire);
}
