from __future__ import annotations

from typing import Dict


def build_evidence_pointer(doc_id: str, page: int, bbox: Dict[str, float]) -> Dict[str, object]:
    return {"doc_id": doc_id, "page": page, "bbox": bbox}
