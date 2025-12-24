# Site-to-PDF Master Guide Web App

This project provides a lightweight web application for crawling a website, summarizing the content, and generating a structured PDF master guide that includes an outline, executive summary, and page-by-page highlights.

## Features

- Crawl multiple pages from a single domain
- Extract headings and paragraphs
- Generate an executive summary and outline
- Download a polished PDF guide

## Requirements

- Python 3.9+

## Setup

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python main.py
```

Then open `http://localhost:5000` in your browser.

## Usage Notes

- Provide a fully-qualified URL (e.g., `https://example.com`).
- Adjust the "Maximum pages to crawl" value to control crawl depth.
