#!/usr/bin/env python3
"""Build site/research.html from site/research.md.

research.md is the single source of truth (Markdown, humanized prose, ```chart NAME```
fences for the JS-rendered charts). This script renders it with python-markdown
(tables + fenced_code + toc) and wraps it in the site shell (site/index.html's nav,
style.css tokens): a dot-matrix "Research" title, a 44px standfirst, and a 1200px
column with a sticky left table of contents. Most element styling (nav, .wrap, h2,
.lede, .code pre, tables, .disclosure-box, .footer) comes straight from style.css —
this script's only job beyond templating is adding the hooks that reuse it
(data-label per td, class="lede" per body paragraph, etc.); research.css holds just
the article-specific layout and the chart-legend classes.

Usage: uv run --with markdown python scripts/build_research.py
"""
import html as html_module
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "site" / "research.md"
OUT = ROOT / "site" / "research.html"

CHART_RE = re.compile(r"```chart\s+(\S+)\s*\n.*?```\n?", re.S)
PARA_RE = re.compile(r"<p>.*?</p>", re.S)
TABLE_RE = re.compile(r"<table>.*?</table>", re.S)
TH_RE = re.compile(r"<th>(.*?)</th>", re.S)
TR_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
TD_RE = re.compile(r"<td>(.*?)</td>", re.S)
CODE_RE = re.compile(r"<pre><code[^>]*>(.*?)</code></pre>", re.S)

CHECKPOINTS = [
    ("typical-small", "https://huggingface.co/OzLabs/typical-small"),
    ("typical-medium", "https://huggingface.co/OzLabs/typical-medium"),
    ("typical-small-preview", "https://huggingface.co/OzLabs/typical-small-preview"),
]

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Research — Typical</title>
<meta name="description" content="Architecture, data, pipeline, findings, and limitations behind Typical." />
<link href="https://api.fontshare.com/v2/css?f[]=switzer@400,500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css" />
<link rel="stylesheet" href="research.css" />
</head>
<body>

<nav class="nav" aria-label="Site">
  <a class="brand" href="index.html">Typical</a>
  <div class="links">
    <a href="index.html#results">Results</a><a href="index.html#demos">Demos</a><a href="research.html" aria-current="page">Research</a>
    <a href="https://huggingface.co/OzLabs/typical-small/tree/main/inference">Code →</a>
    <a href="https://huggingface.co/OzLabs/typical-small">Hugging Face →</a>
  </div>
</nav>

<header class="page-header wrap">
  <div class="title-dots" id="title-dots" aria-label="Research"></div>
{header}
</header>

<main class="layout wrap">
  <aside class="toc" aria-label="Table of contents">
{toc}
  </aside>
  <article class="content">
{article}
  </article>
</main>

<footer class="footer">
  <div class="wrap">
    <span>Typical is research-grade, released open.</span>
    <span class="links">
{footer_links}
    </span>
  </div>
</footer>

<script type="module" src="js/research.js"></script>
</body>
</html>
"""


def extract_charts(text):
    """Turn ```chart name``` fences into chart mount divs before Markdown runs, so
    they pass through as raw HTML blocks. IDs match what js/research.js expects."""

    def repl(m):
        name = m.group(1)
        return f'<div class="chart" id="chart-{name}" data-chart="{name}"></div>\n'

    return CHART_RE.sub(repl, text)


def add_data_labels(article_html):
    """Tables are style.css's hairline-register component, which stacks to label/value
    rows at <=600px by reading data-label off each <td> (see style.css's `table td::before`
    rule) — markdown's table extension doesn't emit that, so stamp it on here."""

    def process_table(m):
        table_html = m.group(0)
        headers = [re.sub(r"<[^>]+>", "", h).strip() for h in TH_RE.findall(table_html)]

        def process_row(rm):
            cells = TD_RE.findall(rm.group(1))
            if not cells:
                return rm.group(0)
            labeled = "\n".join(
                f'<td data-label="{html_module.escape(headers[i], quote=True)}">{c}</td>'
                for i, c in enumerate(cells)
            )
            return f"<tr>\n{labeled}\n</tr>"

        return TR_RE.sub(process_row, table_html)

    return TABLE_RE.sub(process_table, article_html)


def unwrap_code_blocks(html):
    """style.css's .code pre idiom styles <pre> directly (mono, wrap, hairline top/bottom) —
    drop fenced_code's <code class="language-x"> wrapper and add the .code div."""
    return CODE_RE.sub(lambda m: f'<div class="code"><pre>{m.group(1)}</pre></div>', html)


def add_lede(html):
    """Reuse style.css's .lede (14px, cream-70, 66ch) for every body paragraph. Paragraphs
    inside the D1 blockquote and the *italic* figure captions get overridden back down by
    more specific research.css rules regardless of this class."""
    return PARA_RE.sub(lambda m: m.group(0).replace("<p>", '<p class="lede">', 1), html)


def build():
    src = SRC.read_text()
    src = extract_charts(src)

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc"],
        extension_configs={"toc": {"toc_depth": "2-2"}},
    )
    body = md.convert(src)
    toc = md.toc  # <div class="toc"><ul>...</ul></div>, h2-only

    # Everything before the first h2 (the h1 title + standfirst paragraphs) is the page
    # header; everything from the first h2 on is the article body.
    split = re.split(r"(?=<h2)", body, maxsplit=1)
    header_html, article_html = split if len(split) == 2 else ("", body)

    # Title becomes the dot-matrix mount (js/research.js renders it); the first paragraph
    # is the 44px standfirst, any further header paragraphs read as regular body copy.
    header_html = re.sub(r"<h1[^>]*>.*?</h1>\s*", "", header_html, count=1, flags=re.S)
    paras = PARA_RE.findall(header_html)
    header_out = ""
    if paras:
        header_out += paras[0].replace("<p>", '<p class="standfirst">', 1) + "\n"
        header_out += add_lede("\n".join(paras[1:]))

    article_html = add_data_labels(article_html)
    article_html = unwrap_code_blocks(article_html)
    article_html = article_html.replace("<blockquote>", '<blockquote class="disclosure-box">')
    article_html = add_lede(article_html)

    # md.toc wraps its own <div class="toc"><ul>...; unwrap it into our <aside class="toc">
    # and reuse style.css's .ghost link (12px, champagne on hover/current) for every entry.
    toc_ul = re.sub(r'^<div class="toc">\s*|\s*</div>\s*$', "", toc.strip())
    toc_ul = toc_ul.replace("<a href", '<a class="ghost" href')

    footer_links = "\n".join(
        f'      <a href="{href}">{name}</a>' for name, href in CHECKPOINTS
    )

    html = TEMPLATE.format(header=header_out.strip(), toc=toc_ul, article=article_html.strip(), footer_links=footer_links)
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html)} bytes, {len(CHECKPOINTS)} footer links, toc: {toc.count('<li>')} sections)")


if __name__ == "__main__":
    build()
