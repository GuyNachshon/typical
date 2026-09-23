#!/usr/bin/env python3
"""Build site/research.html from site/research.md, on the v11 Agility lane (designs/AGILITY_DESIGN.md,
site/src.css tokens/components — see PLAN5.md). research.md is the single source of truth (Markdown,
```chart NAME``` fences for the JS-rendered charts); prose is not edited here.

Shape: nav + footer copied verbatim from site/index.html; a warm-white opening `.chapter-head`
(eyebrow "RESEARCH", title, standfirst as `.lede`); each `## section` becomes a `.chapter` with a
`.chapter-head` (eyebrow "0N — Title" left, a short first paragraph or the title itself as the big
`.t-section` statement) and the body laid out in a 64ch prose column, offset right in the same
220px+1fr grid as chapter-head. Tables become `.register`, fenced code becomes `.code`, chart fences
become `.media` panels, the D1 blockquote becomes a `.ucard`/`.note`, and the eight findings become a
`.ucard-grid`. Most component styling (chapter, chapter-head, register, code, media, ucard, dest,
footer, nav) comes straight from site/style.css (built from src.css) — research.css only adds the
chart-legend hooks js/charts.js needs and a couple of layout rules src.css doesn't have a component
for (the 64ch prose column, the h1's mobile size).

Usage: uv run --with markdown python scripts/build_research.py
"""
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "site" / "research.md"
OUT = ROOT / "site" / "research.html"

CHART_RE = re.compile(r"```chart\s+(\S+)\s*\n.*?```\n?", re.S)
TABLE_RE = re.compile(r"<table>.*?</table>", re.S)
TH_RE = re.compile(r"<th>(.*?)</th>", re.S)
TR_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
TD_RE = re.compile(r"<td>(.*?)</td>", re.S)
CODE_RE = re.compile(r'<pre><code(?: class="language-text-(wide|narrow)")?[^>]*>(.*?)</code></pre>', re.S)
TAG_RE = re.compile(r"<[^>]+>")
H2_SPLIT_RE = re.compile(r"(?=<h2)")
H2_RE = re.compile(r'^<h2 id="([^"]*)">(.*?)</h2>', re.S)
LI_RE = re.compile(r"<li>\s*(?:<p>)?(.*?)(?:</p>)?\s*</li>", re.S)
BLOCK_START_RE = re.compile(r'(?=^<(?:h3|p|ul|ol|table|div|figure)\b)', re.M)
# A setpiece breaks the prose column and gets its own full-width row. Match on the opening of the
# class attribute, not the whole value: '<div class="media"' missed 'class="media dark"' and
# silently nested a full-width diagram inside the 64ch column, where it drew its stacked layout.
SETPIECE_PREFIXES = ('<table', '<div class="code', '<div class="media', '<div class="ucard', '<figure class="fig')

# Eyebrow category word per chapter (keyed by the h2's slug id): short and never the title itself
# (the eyebrow used to fall back to the full title, printing every chapter head twice).
EYEBROW_CATEGORY = {
    "one-trunk-read-at-71-depth": "ARCHITECTURE",
    "four-buckets-weighted-against-memorising": "DATA",
    "twelve-thousand-steps-under-20": "TRAINING",
    "the-line-dips-before-it-climbs": "TIMELINE",
    "eight-results-each-with-a-control": "FINDINGS",
    "the-long-state-bug-and-what-comes-next": "IN FLIGHT",
    "limitations": "SCOPE",
    "pip-install-three-lines-of-code": "INSTALL",
}

NAV = """<nav class="nav"><a class="brand" href="index.html">Typical</a><span class="links"><a href="index.html#results">Results</a><a href="index.html#how">How it works</a><a href="index.html#demos">Demos</a><a href="research.html" aria-current="page">Research</a><a href="https://huggingface.co/OzLabs/typical-small">Weights</a></span><a class="cta" href="index.html#tryit">Try it live</a></nav>"""

FOOTER = """<footer class="footer"><div class="page">
  <div class="cols">
    <div><p class="t-card" style="max-width:20ch">Models that decide, not generate.</p><p class="t-eyebrow mt-30" style="color:var(--color-mid)">Weights and inference code on Hugging Face. Training code to follow.</p></div>
    <div class="t-mono"><p class="t-eyebrow mb-18">Models</p><p><a href="https://huggingface.co/OzLabs/typical-small">typical-small</a></p><p><a href="https://huggingface.co/OzLabs/typical-medium">typical-medium</a></p><p><a href="https://huggingface.co/OzLabs/typical-small-preview">typical-small-preview</a></p></div>
    <div class="t-mono"><p class="t-eyebrow mb-18">Site</p><p><a href="index.html#results">Results</a></p><p><a href="index.html#demos">Demos</a></p><p><a href="research.html">Research</a></p><p><a href="research.html#limitations">Limitations</a></p></div>
  </div>
  <div class="rule t-eyebrow" style="color:var(--color-mid)">Every number on this page traces to a file under site/data</div>
</div></footer>"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research — Typical</title>
<meta name="description" content="Architecture, data, pipeline, findings, and limitations behind Typical.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&family=Geist+Mono:wght@400;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
<link rel="stylesheet" href="research.css">
</head>
<body>
{nav}

<header class="page chapter">
  <div class="chapter-head">
    <p class="t-eyebrow">RESEARCH</p>
    <div>
      <h1 class="t-hero">How it was built, what moved, what did not.</h1>
      <p class="lede">{standfirst}</p>
      <p class="note mt-18">{subnote}</p>
    </div>
  </div>
</header>

{chapters}

<section class="page chapter">
  <div class="dest">
    <a href="https://huggingface.co/OzLabs/typical-small"><span class="t-card">Get the weights: 1.7B and 4B, open.</span><span class="t-eyebrow">Hugging Face →</span></a>
    <a href="index.html#results"><span class="t-card">Back to the results.</span><span class="t-eyebrow">Typical →</span></a>
  </div>
</section>

{footer}

<script src="js/vendor/gsap.min.js" defer></script>
<script src="js/vendor/ScrollTrigger.min.js" defer></script>
<script type="module" src="js/research.js"></script>
</body>
</html>
"""


def plain_len(fragment):
    return len(TAG_RE.sub("", fragment).strip())


def extract_charts(text):
    """```chart name``` fences -> a white `.media` panel (30px padding) holding the chart mount
    div js/research.js expects. Raw HTML block, passed through Markdown untouched."""

    def repl(m):
        name = m.group(1)
        return f'<div class="media" style="padding:30px"><div class="chart" id="chart-{name}" data-chart="{name}"></div></div>\n'

    return CHART_RE.sub(repl, text)


def add_data_labels(html):
    """.register (site/src.css) stacks to label/value rows at <=40rem by reading data-label off
    each <td> — markdown's table extension doesn't emit that, so stamp it on, and tag the table
    itself .register (markdown emits a bare <table>)."""

    def process_table(m):
        table_html = m.group(0).replace("<table>", '<table class="register">', 1)
        headers = [re.sub(r"<[^>]+>", "", h).strip() for h in TH_RE.findall(table_html)]

        def process_row(rm):
            cells = TD_RE.findall(rm.group(1))
            if not cells:
                return rm.group(0)
            labeled = "\n".join(f'<td data-label="{headers[i]}">{c}</td>' for i, c in enumerate(cells))
            return f"<tr>\n{labeled}\n</tr>"

        return TR_RE.sub(process_row, table_html)

    return TABLE_RE.sub(process_table, html)


def unwrap_code_blocks(html):
    """.code (src.css) styles <pre> directly (dark 12px mono panel) — drop fenced_code's
    <code class="language-x"> wrapper and add the .code div. ```text-wide is the only ASCII-plate
    fence left; research.css scrolls it horizontally rather than maintaining a narrow twin."""
    return CODE_RE.sub(lambda m: f'<div class="code{" " + m.group(1) if m.group(1) else ""}"><pre>{m.group(2)}</pre></div>', html)


BLOCKQUOTE_RE = re.compile(r"<blockquote>(.*?)</blockquote>", re.S)


def convert_blockquote(html):
    """The D1 disclosure blockquote -> a light-gray .ucard holding a .note. Collapsed onto one
    line: layout_body's block splitter looks for tags at the start of a line, and markdown's
    blockquote output puts its <p> on its own line, which would otherwise split this in two."""

    def repl(m):
        inner = " ".join(m.group(1).split("\n"))
        return f'<div class="ucard" style="padding-top:30px;min-height:0;max-width:72ch"><blockquote class="note">{inner}</blockquote></div>'

    return BLOCKQUOTE_RE.sub(repl, html)


def convert_findings_list(html):
    """The eight numbered findings -> a .ucard-grid: number as mono eyebrow, the bold claim as
    .t-feature, the rest of the item as body text. There is exactly one <ol> in research.md."""
    m = re.search(r"<ol>(.*?)</ol>", html, re.S)
    if not m:
        return html
    items = LI_RE.findall(m.group(1))
    cards = []
    for i, item in enumerate(items, start=1):
        sm = re.match(r"<strong>(.*?)</strong>\s*(.*)", item, re.S)
        claim, rest = (sm.group(1), sm.group(2).strip()) if sm else (item, "")
        rest = rest.lstrip(",").strip()
        cards.append(
            f'<div class="ucard" style="padding-top:30px;min-height:0">'
            f'<p class="t-eyebrow muted">{i:02d}</p>'
            f'<p class="t-feature">{claim}</p>'
            f'<p>{rest}</p></div>'
        )
    grid = f'<div class="ucard-grid">{"".join(cards)}</div>'
    return html[: m.start()] + grid + html[m.end() :]


def style_h3(html):
    return re.sub(r"<h3([^>]*)>", r'<h3 class="t-feature"\1>', html)


def layout_body(html):
    """Group consecutive prose blocks (h3/p/ul/ol) into a 64ch `.prose` column; tables, code,
    chart media panels, and ucards are set-pieces that run the full width of the content column."""
    blocks = [b for b in BLOCK_START_RE.split(html) if b.strip()]
    out, prose = [], []

    def flush():
        if prose:
            out.append('<div class="prose">' + "\n".join(prose) + "</div>")
            prose.clear()

    for b in blocks:
        b = b.strip("\n")
        if b.lstrip().startswith(SETPIECE_PREFIXES):
            flush()
            out.append(b)
        else:
            prose.append(b)
    flush()
    return "\n".join(out)


def build_chapters(article_html):
    sections = [s for s in H2_SPLIT_RE.split(article_html) if s.strip()]
    chapters = []
    for i, section in enumerate(sections, start=1):
        hm = H2_RE.match(section)
        title = TAG_RE.sub("", hm.group(2)).strip()
        rest = section[hm.end() :]

        first_block_m = BLOCK_START_RE.search(rest[1:])
        first_block = rest[: (first_block_m.start() + 1) if first_block_m else len(rest)].strip()
        is_para = first_block.startswith("<p>")
        if is_para and plain_len(first_block) < 140:
            headline = first_block[3:-4] if first_block.endswith("</p>") else TAG_RE.sub("", first_block)
            rest = rest[len(first_block) :]
        else:
            headline = title

        body = layout_body(rest.strip())
        category = EYEBROW_CATEGORY.get(hm.group(1), title.split()[0].upper())
        chapters.append(
            f'<section class="page chapter" id="{hm.group(1)}">\n'
            f'  <div class="chapter-head">\n'
            f'    <p class="t-eyebrow">{i:02d} — {category}</p>\n'
            f'    <div><h2 class="t-section">{headline}</h2></div>\n'
            f"  </div>\n"
            f'  <div class="chapter-body"><div class="body">\n{body}\n</div></div>\n'
            f"</section>"
        )
    return "\n\n".join(chapters)


def build():
    src = SRC.read_text()
    src = extract_charts(src)

    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"], extension_configs={"toc": {"toc_depth": "2-2"}})
    body = md.convert(src)

    split = re.split(r"(?=<h2)", body, maxsplit=1)
    header_html, article_html = split if len(split) == 2 else ("", body)
    header_html = re.sub(r"<h1[^>]*>.*?</h1>\s*", "", header_html, count=1, flags=re.S)
    paras = re.findall(r"<p>.*?</p>", header_html, re.S)
    standfirst = TAG_RE.sub("", paras[0]) if paras else ""
    subnote = paras[1][3:-4] if len(paras) > 1 else ""

    article_html = add_data_labels(article_html)
    article_html = unwrap_code_blocks(article_html)
    article_html = convert_blockquote(article_html)
    article_html = convert_findings_list(article_html)
    article_html = style_h3(article_html)

    chapters = build_chapters(article_html)

    html = TEMPLATE.format(
        nav=NAV,
        standfirst=standfirst,
        subnote=subnote,
        chapters=chapters,
        footer=FOOTER,
    )
    OUT.write_text(html)
    n_chapters = chapters.count('class="chapter-head"')
    print(f"wrote {OUT} ({len(html)} bytes, {n_chapters} chapters)")


if __name__ == "__main__":
    build()
