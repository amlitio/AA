from __future__ import annotations

from typing import Dict


class XeroClient:
    def get_invoice(self, tenant_id: str, invoice_id: str) -> Dict[str, object]:
        return {
            "tenant_id": tenant_id,
            "invoice_id": invoice_id,
            "invoice_number": invoice_id,
            "total": 0,
            "date": None,
            "contact": {"name": None},
        }
