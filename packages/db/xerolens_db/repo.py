from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256

from sqlalchemy.orm import Session

from .models import AgentAction, AuditLog, Case


class CaseRepository:
    def __init__(self, session: Session):
        self.session = session

    def upsert_case(self, tenant_id: str, xero_transaction_id: str, risk: int, confidence: float, summary: str = "") -> Case:
        case = (
            self.session.query(Case)
            .filter(Case.tenant_id == tenant_id, Case.xero_transaction_id == xero_transaction_id)
            .one_or_none()
        )
        if case is None:
            case = Case(tenant_id=tenant_id, xero_transaction_id=xero_transaction_id)
            self.session.add(case)

        case.risk_score = risk
        case.confidence = confidence
        case.summary = summary
        case.updated_at = datetime.now(timezone.utc)
        self.session.flush()
        return case

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
