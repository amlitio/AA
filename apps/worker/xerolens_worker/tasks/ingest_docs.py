from __future__ import annotations


def ingest_document_event(event: dict) -> dict:
    return {"status": "ingested", "event": event}
