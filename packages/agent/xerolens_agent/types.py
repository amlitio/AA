from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class Mode(str, Enum):
    ASSIST = "ASSIST"
    AUTOPILOT = "AUTOPILOT"
    FORENSIC = "FORENSIC"


@dataclass
class Evidence:
    kind: str
    ref: str
    detail: Dict[str, Any]


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    data: Dict[str, Any]


@dataclass
class Action:
    action_type: str
    payload: Dict[str, Any]
    confidence: float
    requires_approval: bool
    reason: str
