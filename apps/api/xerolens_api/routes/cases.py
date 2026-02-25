from __future__ import annotations

from typing import Dict, List


def list_cases(tenant_id: str) -> List[Dict[str, str]]:
    return [{"tenant_id": tenant_id, "case_id": "placeholder", "status": "OPEN"}]
