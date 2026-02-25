from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Policy:
    autopilot_enabled: bool = False
    min_match_confidence: float = 0.92
    amount_tolerance: float = 0.50
    require_field_ticket: bool = True
    require_approval_form: bool = False
    forbid_auto_posting: bool = True
    forbid_auto_void: bool = True
