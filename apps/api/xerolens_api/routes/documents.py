from __future__ import annotations

from typing import Dict


def register_document(tenant_id: str, storage_url: str, sha256: str) -> Dict[str, str]:
    return {"tenant_id": tenant_id, "storage_url": storage_url, "sha256": sha256}
