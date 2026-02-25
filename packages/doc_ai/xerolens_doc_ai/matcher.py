from __future__ import annotations

from typing import Dict, List


def match_invoice_to_documents(invoice: Dict[str, object], docs: List[Dict[str, object]]) -> Dict[str, object]:
    """Simple matcher scaffold; upgrade with weighted field-level matching."""
    if not docs:
        return {"confidence": 0.0, "evidence": []}

    best_doc = docs[0]
    confidence = 0.95 if best_doc.get("total") == invoice.get("total") else 0.6
    evidence = [{"kind": "DOC", "ref": best_doc.get("doc_id", "unknown"), "detail": best_doc}]
    return {"confidence": confidence, "evidence": evidence}
