from __future__ import annotations

from typing import Dict, List

from xerolens_agent.decision import decide
from xerolens_agent.policy import Policy
from xerolens_agent.types import Evidence, Finding


class InvoiceCaseWorkflow:
    def __init__(self, policy: Policy, tools):
        self.policy = policy
        self.tools = tools

    def run(self, tenant_id: str, invoice_id: str, mode: str) -> Dict[str, object]:
        invoice = self.tools.xero.get_invoice(tenant_id, invoice_id)

        candidates = self.tools.docs.find_candidates(
            tenant_id=tenant_id,
            amount=invoice["total"],
            contact_name=invoice.get("contact", {}).get("name"),
            invoice_number=invoice.get("invoice_number"),
            date=invoice.get("date"),
        )

        extracted = [self.tools.docs.extract(tenant_id, c["doc_id"]) for c in candidates]
        match = self.tools.matcher.match(invoice, extracted)
        match_conf = match["confidence"]
        evidence: List[Evidence] = match["evidence"]

        findings: List[Finding] = self.tools.validator.validate(invoice, match)
        risk, actions = decide(self.policy, match_conf, findings, evidence)

        case_id = self.tools.db.upsert_case(tenant_id, invoice_id, risk, match_conf, findings, evidence)
        self.tools.db.save_actions(case_id, actions)

        executed = []
        for action in actions:
            if mode == "AUTOPILOT" and not action.requires_approval:
                self.tools.xero.execute_action(tenant_id, invoice_id, action)
                self.tools.db.mark_action_executed(case_id, action)
                executed.append(action.action_type)

        return {
            "case_id": case_id,
            "risk": risk,
            "confidence": match_conf,
            "executed": executed,
        }
