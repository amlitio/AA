from __future__ import annotations

import io
import re
import textwrap
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
from flask import Flask, Response, render_template_string, request, send_file

APP_TITLE = "Site-to-PDF Master Guide"
DEFAULT_MAX_PAGES = 20
DEFAULT_TIMEOUT = 10

app = Flask(__name__)


@dataclass
class PageContent:
    url: str
    title: str
    headings: list[str]
    paragraphs: list[str]


class CrawlResult:
    def __init__(self, pages: list[PageContent]):
        self.pages = pages

    def summary(self) -> str:
        blocks: list[str] = []
        for page in self.pages:
            if page.paragraphs:
                blocks.append(page.paragraphs[0])
            elif page.headings:
                blocks.append(page.headings[0])
        return summarize_text(" ".join(blocks), max_sentences=6)

    def outline(self) -> list[str]:
        outline_items: list[str] = []
        for page in self.pages:
            for heading in page.headings[:5]:
                outline_items.append(f"{page.title}: {heading}")
        return outline_items


PDF_CACHE: dict[str, bytes] = {}


def normalize_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    cleaned = parsed._replace(fragment="", query="")
    return urlunparse(cleaned)


def same_domain(seed_url: str, target_url: str) -> bool:
    return urlparse(seed_url).netloc == urlparse(target_url).netloc


def extract_content(html: str, base_url: str) -> PageContent:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = (soup.title.string if soup.title else base_url).strip()
    headings = [
        normalize_whitespace(tag.get_text())
        for tag in soup.find_all(["h1", "h2", "h3"])
        if normalize_whitespace(tag.get_text())
    ]
    paragraphs = [
        normalize_whitespace(tag.get_text())
        for tag in soup.find_all("p")
        if normalize_whitespace(tag.get_text())
    ]
    return PageContent(url=base_url, title=title, headings=headings, paragraphs=paragraphs)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def summarize_text(text: str, max_sentences: int = 5) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = [s.strip() for s in sentences if s.strip()]
    return " ".join(cleaned[:max_sentences])


def discover_links(soup: BeautifulSoup, base_url: str) -> Iterable[str]:
    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        if href.startswith("#") or href.lower().startswith("mailto:"):
            continue
        absolute = urljoin(base_url, href)
        yield normalize_url(absolute)


def crawl_site(seed_url: str, max_pages: int = DEFAULT_MAX_PAGES) -> CrawlResult:
    visited: set[str] = set()
    queue = deque([normalize_url(seed_url)])
    pages: list[PageContent] = []

    while queue and len(pages) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            continue

        page = extract_content(response.text, url)
        pages.append(page)

        soup = BeautifulSoup(response.text, "html.parser")
        for link_url in discover_links(soup, url):
            if link_url not in visited and same_domain(seed_url, link_url):
                queue.append(link_url)

    return CrawlResult(pages=pages)


def build_pdf(result: CrawlResult, seed_url: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 10, APP_TITLE)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 8, f"Source: {seed_url}")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, "Executive Summary")
    pdf.set_font("Helvetica", "", 12)
    summary_text = result.summary() or "No summary available."
    pdf.multi_cell(0, 8, summary_text)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, "Outline")
    pdf.set_font("Helvetica", "", 12)
    outline_items = result.outline() or ["No outline data available."]
    for item in outline_items:
        pdf.multi_cell(0, 8, f"• {item}")
    pdf.ln(4)

    for page in result.pages:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, page.title or page.url)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, page.url)
        pdf.ln(2)

        if page.headings:
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 6, "Key Headings")
            pdf.set_font("Helvetica", "", 11)
            for heading in page.headings[:10]:
                pdf.multi_cell(0, 6, f"• {heading}")
            pdf.ln(2)

        if page.paragraphs:
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 6, "Page Summary")
            pdf.set_font("Helvetica", "", 11)
            summary = summarize_text(" ".join(page.paragraphs), max_sentences=8)
            wrapped = textwrap.fill(summary or "No paragraph content found.", 120)
            pdf.multi_cell(0, 6, wrapped)
            pdf.ln(2)

            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 6, "Content Highlights")
            pdf.set_font("Helvetica", "", 10)
            for paragraph in page.paragraphs[:6]:
                pdf.multi_cell(0, 5, f"- {paragraph}")
                pdf.ln(1)

    return bytes(pdf.output(dest="S"))


FORM_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>{{ title }}</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; color: #222; }
      .container { max-width: 900px; margin: 0 auto; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
      label { font-weight: bold; display: block; margin-bottom: 0.5rem; }
      input[type=text], input[type=number] { width: 100%; padding: 0.6rem; margin-bottom: 1rem; }
      button { background: #1b4dff; color: #fff; border: none; padding: 0.7rem 1.4rem; border-radius: 6px; cursor: pointer; }
      button:hover { background: #163ed1; }
      .summary { white-space: pre-line; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>{{ title }}</h1>
      <p>Scrape a website, summarize its content, and download a structured PDF master guide.</p>
      <div class="card">
        <form method="post" action="/scrape">
          <label for="url">Website URL</label>
          <input id="url" name="url" type="text" placeholder="https://example.com" required />
          <label for="max_pages">Maximum pages to crawl</label>
          <input id="max_pages" name="max_pages" type="number" min="1" max="100" value="{{ max_pages }}" />
          <button type="submit">Generate Guide</button>
        </form>
      </div>
      {% if error %}
        <div class="card"><strong>Error:</strong> {{ error }}</div>
      {% endif %}
      {% if summary %}
        <div class="card">
          <h2>Executive Summary</h2>
          <p class="summary">{{ summary }}</p>
          <h3>Outline Highlights</h3>
          <ul>
            {% for item in outline %}
              <li>{{ item }}</li>
            {% endfor %}
          </ul>
          <a href="/download/{{ job_id }}">Download PDF</a>
        </div>
      {% endif %}
    </div>
  </body>
</html>
"""


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template_string(
        FORM_TEMPLATE,
        title=APP_TITLE,
        max_pages=DEFAULT_MAX_PAGES,
        summary=None,
        outline=None,
        error=None,
        job_id=None,
    )


@app.route("/scrape", methods=["POST"])
def scrape() -> str:
    seed_url = request.form.get("url", "").strip()
    max_pages = request.form.get("max_pages", str(DEFAULT_MAX_PAGES))

    try:
        max_pages_int = int(max_pages)
    except ValueError:
        max_pages_int = DEFAULT_MAX_PAGES

    if not seed_url:
        return render_template_string(
            FORM_TEMPLATE,
            title=APP_TITLE,
            max_pages=max_pages_int,
            summary=None,
            outline=None,
            error="Please provide a valid URL.",
            job_id=None,
        )

    try:
        result = crawl_site(seed_url, max_pages=max_pages_int)
        pdf_bytes = build_pdf(result, seed_url)
        job_id = str(uuid.uuid4())
        PDF_CACHE[job_id] = pdf_bytes
    except Exception as exc:  # noqa: BLE001
        return render_template_string(
            FORM_TEMPLATE,
            title=APP_TITLE,
            max_pages=max_pages_int,
            summary=None,
            outline=None,
            error=str(exc),
            job_id=None,
        )

    return render_template_string(
        FORM_TEMPLATE,
        title=APP_TITLE,
        max_pages=max_pages_int,
        summary=result.summary(),
        outline=result.outline(),
        error=None,
        job_id=job_id,
    )


@app.route("/download/<job_id>", methods=["GET"])
def download(job_id: str) -> Response:
    pdf_bytes = PDF_CACHE.get(job_id)
    if not pdf_bytes:
        return Response("PDF not found or expired.", status=404)
    return send_file(
        path_or_file=io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="site-master-guide.pdf",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
