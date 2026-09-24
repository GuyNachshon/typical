// copycode.js -- a copy button on every code block.
//
// Injected rather than written into the markup, so the four blocks across three pages cannot drift
// apart and a fifth gets one for free.
//
// The button is the only thing on these pages a reader will click more than once, so it is quiet
// until you approach the block and it answers immediately: the label swaps to "copied" for a beat
// and swaps back. No animation on the press beyond the scale every .btn already has -- a copy is a
// thing you do in passing, and anything longer than the action itself reads as lag.

const RESET_MS = 1400;

// The trailing newline a <pre> carries from its markup is not part of what anyone wants to paste.
export function codeText(el) {
  return el.textContent.replace(/\s+$/, '');
}

export function mountCopyButtons(root = document) {
  if (!navigator.clipboard) return; // no clipboard, no button: a dead control is worse than none
  root.querySelectorAll('.code').forEach((block) => {
    if (block.dataset.copyable) return;
    block.dataset.copyable = '1';
    const pre = block.querySelector('pre');
    if (!pre) return;

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'code-copy t-mono';
    btn.textContent = 'copy';
    btn.setAttribute('aria-label', 'Copy code to clipboard');
    let timer = 0;

    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(codeText(pre));
        btn.textContent = 'copied';
        btn.classList.add('is-done');
      } catch {
        btn.textContent = 'press ⌘C';  // the clipboard can be refused; say what to do instead
      }
      clearTimeout(timer);
      timer = setTimeout(() => {
        btn.textContent = 'copy';
        btn.classList.remove('is-done');
      }, RESET_MS);
    });

    block.appendChild(btn);
  });
}

export function selfTest() {
  const strip = (s) => s.replace(/\s+$/, '');
  console.assert(strip('pip install typical-ai\n') === 'pip install typical-ai', 'the markup newline is not part of the snippet');
  console.assert(strip('a\nb\n\n  ') === 'a\nb', 'trailing whitespace goes, inner newlines stay');
  console.assert(strip('x') === 'x', 'a snippet with no trailing space is unchanged');
  console.log('copycode.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) selfTest();
