#!/usr/bin/env python3
"""Build site/research.html from site/research.md.

research.md is the single source of truth (Markdown, humanized prose, ```chart NAME```
fences for the JS-rendered charts). This script renders it with python-markdown
(tables + fenced_code + toc) and wraps it in the Atoms shell: fixed nav, a 44px title,
and a 1200px column with a sticky left table of contents.

Usage: uv run --with markdown python scripts/build_research.py
"""
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "site" / "research.md"
OUT = ROOT / "site" / "research.html"

CHART_RE = re.compile(r"```chart\s+(\S+)\s*\n.*?```\n?", re.S)

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
<link rel="stylesheet" href="style.css" />
<link rel="stylesheet" href="research.css" />
</head>
<body>

<nav class="nav">
  <div class="nav-inner">
    <a class="wordmark" href="index.html">Typical</a>
    <div class="nav-links">
      <a href="index.html#results">Results</a>
      <a href="index.html#demos">Demos</a>
      <a href="research.html" aria-current="page">Research</a>
      <a href="https://huggingface.co/OzLabs/typical-small/tree/main/inference" target="_blank" rel="noopener">Code &rarr;</a>
      <a href="https://huggingface.co/OzLabs" target="_blank" rel="noopener">Hugging Face &rarr;</a>
    </div>
  </div>
</nav>

<header class="page-header">
{header}
</header>

<main class="layout">
  <aside class="toc" aria-label="Table of contents">
{toc}
  </aside>
  <article class="content">
{article}
  </article>
</main>

<footer class="site-footer">
  <div>Typical is research-grade, released open.</div>
  <div class="links">
{footer_links}
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
        return f'<div class="chart chart-wide" id="chart-{name}" data-chart="{name}"></div>\n'

    return CHART_RE.sub(repl, text)


def build():
    src = SRC.read_text()
    src = extract_charts(src)

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc"],
        extension_configs={"toc": {"toc_depth": "2-2"}},
    )
    body = md.convert(src)
    toc = md.toc  # <div class="toc"><ul>...</ul></div>, h2-only

    # Everything before the first h2 (the h1 title + standfirst paragraphs) is the
    # page header; everything from the first h2 on is the article body.
    split = re.split(r"(?=<h2)", body, maxsplit=1)
    header_html, article_html = split if len(split) == 2 else ("", body)

    # wrap tables so wide ones (JevBench context) scroll instead of blowing out the page
    article_html = article_html.replace("<table>", '<div class="table-scroll"><table>').replace(
        "</table>", "</table></div>"
    )

    footer_links = "\n".join(
        f'    <a class="link-ghost" href="{href}" target="_blank" rel="noopener">{name}</a>'
        for name, href in CHECKPOINTS
    )

    html = TEMPLATE.format(header=header_html.strip(), toc=toc, article=article_html.strip(), footer_links=footer_links)
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html)} bytes, {len(CHECKPOINTS)} footer links, toc: {toc.count('<li>')} sections)")


if __name__ == "__main__":
    build()
