// typography.js — the wrap rules CSS can't express.
//
// `text-wrap: pretty` (set on body) fixes single-word orphans, but it has nothing to say about
// the separators this page uses everywhere: "45 ms / decision · CLINC-150 80.4% / 84.7% · …"
// wraps happily after a "·", leaving a line that ends on a dangling bullet. Binding each
// separator to the word that follows it moves the break one word earlier, so a line ends on
// content and the next line opens with the separator — the way a run-in list should read.
//
// Also glues the "N / M" and "N% / M%" pairs the results copy is full of, so a pair never
// splits across two lines.

const NB = ' ';

function glue(text) {
  return text
    .replace(/·\s+/g, `·${NB}`) // bullet binds forward
    .replace(/([\d%])\s+\/\s+(?=[\d.])/g, `$1${NB}/${NB}`) // "80.4% / 84.7%" stays one unit
    .replace(/ \/ /g, ` /${NB}`); // any other " / " binds forward too (literal spaces only: the
    // number rule above has already replaced its own spaces with NBSPs and must not be undone)
}

const SKIP = new Set(['PRE', 'CODE', 'SCRIPT', 'STYLE', 'TEXTAREA', 'INPUT', 'CANVAS', 'SVG']);

// Every text node in the document except the ones where the exact characters matter (code
// samples, the ASCII plates, form fields). Runs once after mount; anything rendered later glues
// its own strings at the source instead (see shell.js).
export function glueSeparators(root = document.body) {
  if (!root) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(n) {
      for (let el = n.parentElement; el; el = el.parentElement) {
        if (SKIP.has(el.tagName)) return NodeFilter.FILTER_REJECT;
      }
      return n.nodeValue.includes('·') || n.nodeValue.includes('/') ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach((n) => {
    const next = glue(n.nodeValue);
    if (next !== n.nodeValue) n.nodeValue = next;
  });
}

// For strings built at runtime (HUD lines, replay metadata): glue before they hit the DOM.
export const glued = glue;

export function selfTest() {
  console.assert(glue('a · b') === `a ·${NB}b`, 'bullet binds to the following word');
  console.assert(glue('83.6% / 87.4%') === `83.6%${NB}/${NB}87.4%`, 'number pairs stay together');
  console.assert(glue('inference / typical') === `inference /${NB}typical`, 'a slash binds to the word after it');
  console.assert(glue('a · b') !== 'a · b', 'the bullet rule actually fires');
  console.log('typography.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}
