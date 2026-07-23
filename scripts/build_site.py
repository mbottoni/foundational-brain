#!/usr/bin/env python
"""Wrap the Artifact explainer body into a standalone GitHub Pages page.

``docs/explainer.html`` is authored as an Artifact *body* — no <!doctype>,
<head> or <body>, because the Artifact runtime supplies those. GitHub Pages
needs a complete standalone document, so this lifts the <title> and <style>
into a real <head>, adds the meta/Open-Graph tags a shared link wants, and
emits ``docs/index.html``. One source of truth (the explainer), two targets
(the Artifact and the Pages site).

    python scripts/build_site.py
"""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "docs" / "explainer.html"
OUT = REPO_ROOT / "docs" / "index.html"

DESCRIPTION = (
    "A layman-friendly, animated walkthrough of a self-supervised fMRI "
    "foundation model — what it predicts, and honestly what it did and didn't "
    "learn. Every number is measured, not illustrative."
)


def brain_favicon() -> str:
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
        "<text y='.9em' font-size='90'>🧠</text></svg>"
    )
    return "data:image/svg+xml," + urllib.parse.quote(svg)


def build() -> str:
    html = SRC.read_text()

    m_title = re.search(r"<title>(.*?)</title>", html, re.S)
    title = m_title.group(1).strip() if m_title else "foundational-brain"
    if m_title:
        html = html[: m_title.start()] + html[m_title.end():]

    m_style = re.search(r"<style>.*?</style>", html, re.S)
    style = m_style.group(0) if m_style else ""
    if m_style:
        html = html[: m_style.start()] + html[m_style.end():]

    body = html.strip()

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="description" content="{DESCRIPTION}">
<meta property="og:type" content="website">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{DESCRIPTION}">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="{brain_favicon()}">
<title>{title}</title>
{style}
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    OUT.write_text(build())
    kb = OUT.stat().st_size // 1024
    print(f"wrote {OUT} ({kb} KB)")


if __name__ == "__main__":
    main()
