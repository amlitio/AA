from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Iterable, Protocol

INVOICED_STATUSES = ("DRAFT", "SUBMITTED", "AUTHORISED", "PAID")


class RevenueRepo(Protocol):
    def get_field_tickets(self, tenant_id: str): ...
    def get_invoice_index(self, tenant_id: str, statuses: tuple[str, ...]): ...
    def get_invoice_duplicates(self, tenant_id: str, statuses: tuple[str, ...]): ...
    def get_amount_mismatches(self, tenant_id: str, tolerance: Decimal, statuses: tuple[str, ...]): ...
    def create_case(self, **kwargs): ...


@dataclass
class RevenueIntegrityReport:
    revenue_leakage: int
    unsupported_billing: int
    duplicates: int
    amount_mismatch: int
    total_at_risk: Decimal


def _sum_at_risk(
    missing_numbers: Iterable[str],
    orphan_numbers: Iterable[str],
    amount_mismatches: list[tuple[str, Decimal, Decimal]],
    ticket_totals: dict[str, Decimal],
    invoice_totals: dict[str, Decimal],
) -> Decimal:
    risk = Decimal("0")
    for ft in missing_numbers:
        risk += ticket_totals.get(ft, Decimal("0"))
    for inv in orphan_numbers:
        risk += invoice_totals.get(inv, Decimal("0"))
    for _, packet_total, invoice_total in amount_mismatches:
        risk += abs(packet_total - invoice_total)
    return risk


def reconcile_revenue(
    tenant_id: str,
    repo: RevenueRepo,
    statuses: tuple[str, ...] = INVOICED_STATUSES,
    amount_tolerance: Decimal = Decimal("0.50"),
) -> dict[str, object]:
    tickets = repo.get_field_tickets(tenant_id)
    invoices = repo.get_invoice_index(tenant_id, statuses=statuses)

    ticket_set = {ticket.field_ticket_number for ticket in tickets}
    invoice_set = {invoice.invoice_number for invoice in invoices}

    ticket_totals = {ticket.field_ticket_number: Decimal(ticket.packet_total) for ticket in tickets}
    invoice_totals = {invoice.invoice_number: Decimal(invoice.total) for invoice in invoices}

    missing = ticket_set - invoice_set
    orphan = invoice_set - ticket_set

    for ft in sorted(missing):
        repo.create_case(
            tenant_id=tenant_id,
            case_type="REVENUE_LEAKAGE",
            reference=ft,
            status="READY_FOR_CONFIRMATION",
            summary="Field ticket exists with no invoice in invoiced statuses.",
        )

    for inv in sorted(orphan):
        repo.create_case(
            tenant_id=tenant_id,
            case_type="UNSUPPORTED_BILLING",
            reference=inv,
            status="OPEN",
            summary="Invoice exists in Xero but no matching field ticket packet was found.",
        )

    duplicates = repo.get_invoice_duplicates(tenant_id, statuses=statuses)
    for invoice_number, duplicate_count in sorted(duplicates.items()):
        repo.create_case(
            tenant_id=tenant_id,
            case_type="DUPLICATE",
            reference=invoice_number,
            status="OPEN",
            summary=f"Invoice number appears {duplicate_count} times in invoice index.",
        )

    mismatches = repo.get_amount_mismatches(tenant_id, tolerance=amount_tolerance, statuses=statuses)
    for invoice_number, packet_total, invoice_total in mismatches:
        repo.create_case(
            tenant_id=tenant_id,
            case_type="AMOUNT_MISMATCH",
            reference=invoice_number,
            status="ESCALATED",
            summary=(
                f"Packet total {packet_total} does not match Xero total {invoice_total} "
                f"within tolerance {amount_tolerance}."
            ),
        )

    total_at_risk = _sum_at_risk(missing, orphan, mismatches, ticket_totals, invoice_totals)
    report = RevenueIntegrityReport(
        revenue_leakage=len(missing),
        unsupported_billing=len(orphan),
        duplicates=sum(duplicates.values()),
        amount_mismatch=len(mismatches),
        total_at_risk=total_at_risk,
    )
    return {
        "tenant_id": tenant_id,
        "invoiced_statuses": list(statuses),
        "report": {
            **asdict(report),
            "total_at_risk": str(report.total_at_risk),
        },
    }
