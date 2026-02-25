from __future__ import annotations


def process_webhook(event: dict) -> dict:
    return {"status": "processed", "event": event}
