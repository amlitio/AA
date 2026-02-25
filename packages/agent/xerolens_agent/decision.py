from __future__ import annotations

from typing import List, Tuple

from .policy import Policy
from .scoring import compute_risk
from .types import Action, Evidence, Finding


SAFE_AUTOPILOT_ACTIONS = {"ATTACH_DOCUMENTS", "SET_REFERENCE_FIELDS", "NORMALIZE_CONTACT_METADATA"}


def decide(
    policy: Policy,
    match_conf: float,
    findings: List[Finding],
    evidence: List[Evidence],
) -> Tuple[int, List[Action]]:
    risk = compute_risk(findings)

    actions: List[Action] = []
    has_high = any(f.severity == "HIGH" for f in findings)

    doc_ids = [e.ref for e in evidence if e.kind in {"DOC", "DOC_PAGE", "FIELD_TICKET"}]
    if doc_ids:
        actions.append(
            Action(
                action_type="ATTACH_DOCUMENTS",
                payload={"doc_ids": doc_ids},
                confidence=match_conf,
                requires_approval=False,
                reason="Attach matched backup documents to strengthen audit trail.",
            )
        )

    can_autopilot = (
        policy.autopilot_enabled
        and match_conf >= policy.min_match_confidence
        and not has_high
        and risk <= 30
    )

    for action in actions:
        action.requires_approval = not (can_autopilot and action.action_type in SAFE_AUTOPILOT_ACTIONS)

    return risk, actions
