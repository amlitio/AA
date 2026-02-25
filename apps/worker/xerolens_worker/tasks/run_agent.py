from __future__ import annotations


def run_invoice_case(tenant_id: str, invoice_id: str, mode: str = "ASSIST") -> dict:
    return {
        "tenant_id": tenant_id,
        "invoice_id": invoice_id,
        "mode": mode,
        "status": "queued",
    }


def daily_close(tenant_id: str) -> dict:
    return {"tenant_id": tenant_id, "status": "daily_close_started"}
