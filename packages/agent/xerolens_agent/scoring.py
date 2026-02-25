"""Risk scoring utilities for agent findings."""

from __future__ import annotations

from typing import Iterable

from .types import Finding

DEFAULT_WEIGHTS = {
    "MISSING_APPROVAL": 25,
    "LOCATION_MISMATCH": 30,
    "AMOUNT_MISMATCH": 40,
    "DUPLICATE_SUSPECT": 35,
    "WEAK_MATCH": 20,
    "MANUAL_OVERRIDE": 15,
}


def compute_risk(findings: Iterable[Finding]) -> int:
    score = sum(DEFAULT_WEIGHTS.get(f.code, 10) for f in findings)
    return min(score, 100)
