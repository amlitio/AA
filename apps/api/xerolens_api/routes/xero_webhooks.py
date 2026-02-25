from __future__ import annotations

import hmac
import hashlib
import base64
from typing import Dict


def verify_xero_signature(raw_body: bytes, signature: str, webhook_key: str) -> bool:
    digest = hmac.new(webhook_key.encode("utf-8"), raw_body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def enqueue_webhook_event(event: Dict[str, object]) -> Dict[str, object]:
    return {"queued": True, "event": event}
