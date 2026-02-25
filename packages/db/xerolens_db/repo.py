from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from typing import Iterable, Sequence

from sqlalchemy.orm import Session

from .models import AgentAction, AuditLog, Case, FieldTicket, InvoiceIndex

INVOICED_STATUSES = ("DRAFT", "SUBMITTED", "AUTHORISED", "PAID")


class CaseRepository:
    def __init__(self, session: Session):
        self.session = session

    def upsert_case(
        self,
        tenant_id: str,
        xero_transaction_id: str,
        risk: int,
        confidence: float,
        summary: str = "",
        case_type: str = "GENERAL",
        reference: str = "",
    ) -> Case:
        case = (
            self.session.query(Case)
            .filter(Case.tenant_id == tenant_id, Case.xero_transaction_id == xero_transaction_id, Case.case_type == case_type)
            .one_or_none()
        )
        if case is None:
            case = Case(
                tenant_id=tenant_id,
                xero_transaction_id=xero_transaction_id,
                case_type=case_type,
                reference=reference,
            )
            self.session.add(case)

        case.risk_score = risk
        case.confidence = confidence
        case.summary = summary
        case.updated_at = datetime.now(timezone.utc)
        self.session.flush()
        return case

    def create_case(self, tenant_id: str, case_type: str, reference: str, status: str, summary: str = "") -> Case:
        case = Case(
            tenant_id=tenant_id,
            xero_transaction_id=reference,
            case_type=case_type,
            reference=reference,
            status=status,
            summary=summary,
        )
        self.session.add(case)
        self.session.flush()
        return case

    def get_field_tickets(self, tenant_id: str) -> Sequence[FieldTicket]:
        return self.session.query(FieldTicket).filter(FieldTicket.tenant_id == tenant_id).all()

    def get_invoice_index(self, tenant_id: str, statuses: Iterable[str] = INVOICED_STATUSES) -> Sequence[InvoiceIndex]:
        return (
            self.session.query(InvoiceIndex)
            .filter(InvoiceIndex.tenant_id == tenant_id, InvoiceIndex.status.in_(tuple(statuses)))
            .all()
        )

    def get_invoice_duplicates(self, tenant_id: str, statuses: Iterable[str] = INVOICED_STATUSES) -> dict[str, int]:
        invoices = self.get_invoice_index(tenant_id, statuses)
        counts: dict[str, int] = {}
        for invoice in invoices:
            counts[invoice.invoice_number] = counts.get(invoice.invoice_number, 0) + 1
        return {k: v for k, v in counts.items() if v > 1}

    def get_amount_mismatches(
        self,
        tenant_id: str,
        tolerance: Decimal = Decimal("0.50"),
        statuses: Iterable[str] = INVOICED_STATUSES,
    ) -> list[tuple[str, Decimal, Decimal]]:
        tickets = {ticket.field_ticket_number: Decimal(ticket.packet_total) for ticket in self.get_field_tickets(tenant_id)}
        invoices = {
            invoice.invoice_number: Decimal(invoice.total)
            for invoice in self.get_invoice_index(tenant_id, statuses)
            if invoice.invoice_number in tickets
        }
        mismatches = []
        for number, packet_total in tickets.items():
            invoice_total = invoices.get(number)
            if invoice_total is None:
                continue
            if abs(packet_total - invoice_total) > tolerance:
                mismatches.append((number, packet_total, invoice_total))
        return mismatches

    def save_action(self, case_id, action) -> AgentAction:
        row = AgentAction(
            case_id=case_id,
            action_type=action.action_type,
            payload_json=action.payload,
            mode="PROPOSED",
            reason=action.reason,
            confidence=action.confidence,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def log_event(self, case_id, event_type: str, event: dict) -> AuditLog:
        payload_hash = sha256(repr(event).encode("utf-8")).hexdigest()
        row = AuditLog(case_id=case_id, event_type=event_type, event_json=event, hash=payload_hash)
        self.session.add(row)
        self.session.flush()
        return row
