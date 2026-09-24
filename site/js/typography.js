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

// Anything that reads as an identifier rather than a word: a path, a filename, a section anchor,
// a run name. Ten characters or more, no spaces, and at least one of / . : # _ in it.
const PATH = /\S*[/.:#_]\S*/g;

function glue(text) {
  return text
    // Paths and filenames are single tokens; the browser still offers a break at every hyphen in
    // them, which is how "releases/typical-small.md#latency" ended up split as "releases/typical-"
    // and "small.md#latency". Non-breaking hyphens keep each path whole.
    // Only up to 36 characters: a long slash-list like probe/jev/bench/zero-shot/smoke/dump/leak
    // still needs its hyphens as break opportunities, or one chunk of it pushes the page sideways.
    .replace(PATH, (m) => (m.length >= 10 && m.length <= 36 ? m.replace(/-/g, '\u2011') : m))
    // Names like typical-small-preview or jev_native_v3t20_wf have no slash or dot but are still
    // identifiers: two or more hyphens, no spaces, all lower case.
    .replace(/(?<![^\s(])[a-z0-9]+(?:-[a-z0-9]+){2,}\b/g, (m) => m.replace(/-/g, '\u2011'))
    // Chrome offers a line break after a closing bracket, so "(… 2.03 → 1.01)." can strand its
    // full stop at the start of the next line. A word joiner closes that gap.
    .replace(/\)(?!\u2060)\.(\s|$)/g, ')\u2060.$1')
    // A full stop after a long identifier must not be left to start a line on its own when the
    // identifier breaks: a word joiner binds it to whatever chunk ends up last.
    .replace(/(\S{10,})(?<!\u2060)\.(\s|$)/g, (m, tok, tail) => (/[/:#_]/.test(tok) ? `${tok}\u2060.${tail}` : m))
    // "Source:" and friends belong to what follows them, not to the end of the previous line.
    .replace(/(\w+:)\s+(?=\S)/g, `$1${NB}`)
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

// Widow control: bind the last two words of a block so a paragraph can never end on a lone word.
// CSS `text-wrap: pretty` is supposed to do this and mostly does, but it gives up in narrow
// columns and on blocks with inline children, which is exactly where the widows showed up.
export function bindWidows(root = document.body) {
  if (!root) return;
  const sel = 'p, li, figcaption, blockquote, h1, h2, h3, h4, .lede, .sub';
  root.querySelectorAll(sel).forEach((el) => {
    if (el.dataset.widowed || el.closest('pre, code, .code')) return;
    el.dataset.widowed = '1';
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    let last = null;
    while (walker.nextNode()) if (walker.currentNode.nodeValue.trim()) last = walker.currentNode;
    if (!last) return;
    // two real words, and neither so long that gluing them would overflow the column
    const m = last.nodeValue.match(/(\S+)(\s+)(\S+)\s*$/);
    if (m) {
      if (m[1].length + m[3].length > 22) return;
      last.nodeValue = last.nodeValue.replace(/(\S+)(\s+)(\S+)(\s*)$/, `$1${NB}$3$4`);
      return;
    }
    // A paragraph ending "…<code>typical-medium</code> (4B)." leaves " (4B)." as its whole final
    // text node: one word, so the pair rule above finds nothing to bind and the word drops to a
    // line of its own. Bind it backwards across the element boundary instead -- the thing before
    // it is the element that made this node one word long.
    const solo = last.nodeValue.match(/^(\s+)(\S+)(\s*)$/);
    if (solo && solo[2].length <= 16) last.nodeValue = `${NB}${solo[2]}${solo[3]}`;
  });
}

export function selfTest() {
  const soloDoc = typeof document !== 'undefined';
  if (soloDoc) {
    const host = document.createElement('p');
    host.innerHTML = 'two open models: <code>typical-small</code> and <code>typical-medium</code> (4B).';
    bindWidows(host.ownerDocument.body.appendChild(host).parentNode);
    console.assert(host.textContent.includes(`${NB}(4B).`), 'a lone word after an inline element binds backwards');
    host.remove();
  }
  console.assert(glue('a · b') === `a ·${NB}b`, 'bullet binds to the following word');
  console.assert(glue('83.6% / 87.4%') === `83.6%${NB}/${NB}87.4%`, 'number pairs stay together');
  console.assert(glue('inference / typical') === `inference /${NB}typical`, 'a slash binds to the word after it');
  console.assert(glue('releases/typical-small.md#latency') === 'releases/typical\u2011small.md#latency', 'a path keeps non-breaking hyphens');
  console.assert(glue('gpu-runpod:REPORT.md#\u00a73ab') === 'gpu\u2011runpod:REPORT.md#\u00a73ab', 'a branch:anchor reference is one token too');
  console.assert(glue('probe/jev/bench/zero-shot/smoke/dump/leak/kb-audit').includes('-'), 'a long slash-list keeps its hyphens as break opportunities');
  console.assert(glue('see REPORT.md#3ab-apples-table.').includes('\u2060.'), 'a full stop after an identifier cannot start a line');
  console.assert(!glue('a normal sentence ends here.').includes('\u2060'), 'ordinary sentences are untouched');
  console.assert(glue('(NLL 2.03 \u2192 1.01). Next') === '(NLL 2.03 \u2192 1.01)\u2060. Next', 'a full stop after a bracket stays put');
  console.assert(glue('well-known trade-off') === 'well-known trade-off', 'ordinary hyphenated words are left alone');
  console.assert(glue('typical-small-preview') === 'typical\u2011small\u2011preview', 'a run name is one token');
  console.assert(glue('gpu-runpod-full-experiment:REPORT.md#3ab-apples-to-apples-table').includes('-'), 'a hyphen run inside a longer reference keeps its break opportunities');
  console.assert(glue(glue('see REPORT.md#3ab-x.')) === glue('see REPORT.md#3ab-x.'), 'gluing twice changes nothing');
  console.assert(glue('Source: runs/ts1b') === `Source:${NB}runs/ts1b`, 'a label binds to what it labels');
  console.assert(glue('a · b') !== 'a · b', 'the bullet rule actually fires');
  console.log('typography.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}
