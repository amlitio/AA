from __future__ import annotations


def sync_tenant(tenant_id: str) -> dict:
    return {"tenant_id": tenant_id, "status": "synced"}
