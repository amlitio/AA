from __future__ import annotations

from typing import Dict, List


def validate_packet(invoice: Dict[str, object], match_result: Dict[str, object]) -> List[Dict[str, object]]:
    findings = []
    if match_result.get("confidence", 0) < 0.92:
        findings.append(
            {
                "code": "WEAK_MATCH",
                "severity": "MED",
                "message": "Document match confidence is below autopilot threshold.",
                "data": {"confidence": match_result.get("confidence", 0)},
            }
        )
    if not match_result.get("evidence"):
        findings.append(
            {
                "code": "MISSING_BACKUP",
                "severity": "HIGH",
                "message": "No backup document evidence found.",
                "data": {"invoice_id": invoice.get("invoice_id")},
            }
        )
    return findings
