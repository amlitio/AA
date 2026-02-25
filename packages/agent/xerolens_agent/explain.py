"""Human-readable and auditable explanations for agent decisions."""

from __future__ import annotations

from typing import List

from .types import Action, Evidence, Finding


def build_explanation(risk: int, findings: List[Finding], actions: List[Action], evidence: List[Evidence]) -> str:
    finding_codes = ", ".join(f.code for f in findings) if findings else "none"
    action_names = ", ".join(a.action_type for a in actions) if actions else "none"
    evidence_refs = ", ".join(e.ref for e in evidence) if evidence else "none"
    return (
        f"risk={risk}; findings={finding_codes}; actions={action_names}; evidence={evidence_refs}."
    )
