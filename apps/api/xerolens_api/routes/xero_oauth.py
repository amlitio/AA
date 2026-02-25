from __future__ import annotations

from typing import Dict


def begin_oauth(flow: str = "auth_code") -> Dict[str, str]:
    if flow not in {"auth_code", "pkce"}:
        raise ValueError("flow must be 'auth_code' or 'pkce'")
    return {"flow": flow, "auth_url": "https://login.xero.com/identity/connect/authorize"}


def oauth_callback(code: str, state: str | None = None, code_verifier: str | None = None) -> Dict[str, str]:
    if not code:
        raise ValueError("missing auth code")
    return {"status": "connected", "code": code, "state": state or "", "code_verifier": code_verifier or ""}
