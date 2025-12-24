from __future__ import annotations

import html
import io
import textwrap
import uuid
from collections import deque
from dataclasses import dataclass
from html.parser import HTMLParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

APP_TITLE = "Site-to-PDF Master Guide"
DEFAULT_MAX_PAGES = 20
DEFAULT_TIMEOUT = 10
PDF_CACHE: dict[str, bytes] = {}


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


class ContentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.headings: list[str] = []
        self.paragraphs: list[str] = []
        self.links: list[str] = []
        self.title: str = ""
        self._current_tag: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2", "h3", "p", "title"}:
            self._current_tag = tag
            self._buffer = []
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.links.append(value)

    def handle_data(self, data: str) -> None:
        if self._current_tag:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self._current_tag:
            return
        text = normalize_whitespace(" ".join(self._buffer))
        if not text:
            self._current_tag = None
            self._buffer = []
            return
        if tag == "title":
            self.title = text
        elif tag in {"h1", "h2", "h3"}:
            self.headings.append(text)
        elif tag == "p":
            self.paragraphs.append(text)
        self._current_tag = None
        self._buffer = []


class SimplePDF:
    def __init__(self, title: str) -> None:
        self.title = title
        self.lines: list[str] = []

    def add_heading(self, text: str) -> None:
        self.lines.append(f"# {text}")

    def add_text(self, text: str) -> None:
        self.lines.extend(textwrap.wrap(text, width=90) or [""])

    def add_bullet(self, text: str) -> None:
        wrapped = textwrap.wrap(text, width=86)
        if not wrapped:
            self.lines.append("•")
            return
        self.lines.append(f"• {wrapped[0]}")
        for line in wrapped[1:]:
            self.lines.append(f"  {line}")

    def add_separator(self) -> None:
        self.lines.append("")

    def render(self) -> bytes:
        pages = paginate_lines(self.lines, max_lines=48)
        return build_basic_pdf(pages, title=self.title)


FORM_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 2rem; color: #222; }}
      .container {{ max-width: 900px; margin: 0 auto; }}
      .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }}
      label {{ font-weight: bold; display: block; margin-bottom: 0.5rem; }}
      input[type=text], input[type=number] {{ width: 100%; padding: 0.6rem; margin-bottom: 1rem; }}
      button {{ background: #1b4dff; color: #fff; border: none; padding: 0.7rem 1.4rem; border-radius: 6px; cursor: pointer; }}
      button:hover {{ background: #163ed1; }}
      .summary {{ white-space: pre-line; }}
    </style>
  </head>
  <body>
    <div class="container">
      <h1>{title}</h1>
      <p>Scrape a website, summarize its content, and download a structured PDF master guide.</p>
      <div class="card">
        <form method="post" action="/scrape">
          <label for="url">Website URL</label>
          <input id="url" name="url" type="text" placeholder="https://example.com" required value="{url_value}" />
          <label for="max_pages">Maximum pages to crawl</label>
          <input id="max_pages" name="max_pages" type="number" min="1" max="100" value="{max_pages}" />
          <button type="submit">Generate Guide</button>
        </form>
      </div>
      {error_block}
      {summary_block}
    </div>
  </body>
</html>
"""


def normalize_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    cleaned = parsed._replace(fragment="", query="")
    return urlunparse(cleaned)


def same_domain(seed_url: str, target_url: str) -> bool:
    return urlparse(seed_url).netloc == urlparse(target_url).netloc


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def summarize_text(text: str, max_sentences: int = 5) -> str:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return ". ".join(sentences[:max_sentences]).strip()


def extract_content(html_text: str, base_url: str) -> PageContent:
    parser = ContentParser()
    parser.feed(html_text)
    title = parser.title or base_url
    return PageContent(
        url=base_url,
        title=title,
        headings=parser.headings,
        paragraphs=parser.paragraphs,
    )


def discover_links(html_text: str) -> list[str]:
    parser = ContentParser()
    parser.feed(html_text)
    return parser.links


def fetch_url(url: str) -> tuple[str, str]:
    request = Request(url, headers={"User-Agent": "SiteMasterGuideBot/1.0"})
    with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
        content_type = response.headers.get("Content-Type", "")
        charset = response.headers.get_content_charset() or "utf-8"
        html_text = response.read().decode(charset, errors="replace")
        return content_type, html_text


def crawl_site(seed_url: str, max_pages: int = DEFAULT_MAX_PAGES) -> CrawlResult:
    visited: set[str] = set()
    queue = deque([normalize_url(seed_url)])
    pages: list[PageContent] = []

    while queue and len(pages) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        content_type, html_text = fetch_url(url)
        if "text/html" not in content_type:
            continue

        page = extract_content(html_text, url)
        pages.append(page)

        for link in discover_links(html_text):
            absolute = normalize_url(urljoin(url, link))
            if absolute not in visited and same_domain(seed_url, absolute):
                queue.append(absolute)

    return CrawlResult(pages=pages)


def build_pdf(result: CrawlResult, seed_url: str) -> bytes:
    pdf = SimplePDF(APP_TITLE)
    pdf.add_heading(APP_TITLE)
    pdf.add_text(f"Source: {seed_url}")
    pdf.add_separator()

    pdf.add_heading("Executive Summary")
    pdf.add_text(result.summary() or "No summary available.")
    pdf.add_separator()

    pdf.add_heading("Outline")
    outline_items = result.outline() or ["No outline data available."]
    for item in outline_items:
        pdf.add_bullet(item)
    pdf.add_separator()

    for page in result.pages:
        pdf.add_heading(page.title or page.url)
        pdf.add_text(page.url)
        if page.headings:
            pdf.add_text("Key Headings")
            for heading in page.headings[:10]:
                pdf.add_bullet(heading)
        if page.paragraphs:
            pdf.add_text("Page Summary")
            summary = summarize_text(" ".join(page.paragraphs), max_sentences=8)
            pdf.add_text(summary or "No paragraph content found.")
            pdf.add_text("Content Highlights")
            for paragraph in page.paragraphs[:6]:
                pdf.add_bullet(paragraph)
        pdf.add_separator()

    return pdf.render()


def paginate_lines(lines: list[str], max_lines: int = 48) -> list[list[str]]:
    pages: list[list[str]] = []
    for i in range(0, len(lines), max_lines):
        pages.append(lines[i : i + max_lines])
    return pages or [[""]]


def build_basic_pdf(pages: list[list[str]], title: str) -> bytes:
    objects: list[str] = []

    def add_object(obj: str) -> int:
        objects.append(obj)
        return len(objects)

    font_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    content_ids: list[int] = []
    for page_lines in pages:
        stream = build_content_stream(page_lines)
        content = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
        content_ids.append(add_object(content))

    pages_id = len(objects) + len(pages) + 1
    page_ids: list[int] = []
    for content_id in content_ids:
        page_obj = (
            "<< /Type /Page /Parent "
            f"{pages_id} 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        )
        page_ids.append(add_object(page_obj))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_obj = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"
    pages_id_actual = add_object(pages_obj)

    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id_actual} 0 R >>")

    pdf_body = io.StringIO()
    pdf_body.write("%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(pdf_body.tell())
        pdf_body.write(f"{index} 0 obj\n{obj}\nendobj\n")

    xref_start = pdf_body.tell()
    pdf_body.write(f"xref\n0 {len(objects) + 1}\n")
    pdf_body.write("0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf_body.write(f"{offset:010d} 00000 n \n")
    pdf_body.write(
        "trailer\n"
        f"<< /Size {len(objects) + 1} /Root {catalog_id} 0 R /Title ({pdf_escape(title)}) >>\n"
        "startxref\n"
        f"{xref_start}\n"
        "%%EOF\n"
    )
    return pdf_body.getvalue().encode("latin-1", errors="replace")


def build_content_stream(lines: list[str]) -> str:
    y_position = 800
    leading = 14
    stream_lines = ["BT", "/F1 12 Tf", f"50 {y_position} Td"]
    for line in lines:
        escaped = pdf_escape(line)
        stream_lines.append(f"({escaped}) Tj")
        stream_lines.append(f"0 -{leading} Td")
    stream_lines.append("ET")
    return "\n".join(stream_lines)


def pdf_escape(text: str) -> str:
    return (
        html.escape(text)
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def render_form(
    *,
    url_value: str = "",
    max_pages: int = DEFAULT_MAX_PAGES,
    summary: str | None = None,
    outline: list[str] | None = None,
    error: str | None = None,
    job_id: str | None = None,
) -> bytes:
    error_block = ""
    if error:
        error_block = f'<div class="card"><strong>Error:</strong> {html.escape(error)}</div>'

    summary_block = ""
    if summary:
        summary_html = html.escape(summary)
        outline_html = "".join(
            f"<li>{html.escape(item)}</li>" for item in (outline or [])
        )
        summary_block = (
            "<div class=\"card\">"
            "<h2>Executive Summary</h2>"
            f"<p class=\"summary\">{summary_html}</p>"
            "<h3>Outline Highlights</h3>"
            f"<ul>{outline_html}</ul>"
            f"<a href=\"/download/{job_id}\">Download PDF</a>"
            "</div>"
        )

    html_body = FORM_TEMPLATE.format(
        title=APP_TITLE,
        url_value=html.escape(url_value),
        max_pages=max_pages,
        error_block=error_block,
        summary_block=summary_block,
    )
    return html_body.encode("utf-8")


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self.respond(render_form())
            return
        if self.path.startswith("/download/"):
            job_id = self.path.split("/download/")[-1]
            pdf_bytes = PDF_CACHE.get(job_id)
            if not pdf_bytes:
                self.send_error(HTTPStatus.NOT_FOUND, "PDF not found or expired.")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/pdf")
            self.send_header(
                "Content-Disposition",
                "attachment; filename=site-master-guide.pdf",
            )
            self.send_header("Content-Length", str(len(pdf_bytes)))
            self.end_headers()
            self.wfile.write(pdf_bytes)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/scrape":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8", errors="replace")
        data = parse_qs(body)
        url_value = data.get("url", [""])[0].strip()
        max_pages_raw = data.get("max_pages", [str(DEFAULT_MAX_PAGES)])[0]

        try:
            max_pages = max(1, min(100, int(max_pages_raw)))
        except ValueError:
            max_pages = DEFAULT_MAX_PAGES

        if not url_value:
            self.respond(
                render_form(
                    url_value=url_value,
                    max_pages=max_pages,
                    error="Please provide a valid URL.",
                )
            )
            return

        try:
            result = crawl_site(url_value, max_pages=max_pages)
            pdf_bytes = build_pdf(result, url_value)
            job_id = str(uuid.uuid4())
            PDF_CACHE[job_id] = pdf_bytes
            self.respond(
                render_form(
                    url_value=url_value,
                    max_pages=max_pages,
                    summary=result.summary(),
                    outline=result.outline(),
                    job_id=job_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.respond(
                render_form(
                    url_value=url_value,
                    max_pages=max_pages,
                    error=str(exc),
                )
            )

    def respond(self, body: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "0.0.0.0", port: int = 5000) -> None:
    server = HTTPServer((host, port), RequestHandler)
    print(f"{APP_TITLE} running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
