from __future__ import annotations

from typing import Dict


def extract_document_fields(raw_text: str) -> Dict[str, object]:
    """Baseline parser for contractor invoice/field-ticket packet fields."""
    return {
        "invoice_reference": None,
        "due_date": None,
        "terms": None,
        "subtotal": None,
        "total": None,
        "line_items": [],
        "field_ticket": {
            "crew": [],
            "equipment": [],
            "locations": [],
            "notes": None,
        },
        "raw_text": raw_text,
    }
