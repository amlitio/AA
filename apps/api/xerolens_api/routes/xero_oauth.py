from __future__ import annotations

from typing import Dict
from urllib.parse import urlencode

from xerolens_api.settings import settings


XERO_AUTHORIZE_URL = "https://login.xero.com/identity/connect/authorize"


def begin_oauth(flow: str = "auth_code", state: str = "", code_challenge: str | None = None) -> Dict[str, str]:
    if flow not in {"auth_code", "pkce"}:
        raise ValueError("flow must be 'auth_code' or 'pkce'")

    query = {
        "response_type": "code",
        "client_id": settings.xero_client_id,
        "redirect_uri": settings.xero_redirect_uri,
        "scope": " ".join(settings.xero_scopes),
        "state": state,
    }
    if flow == "pkce":
        if not code_challenge:
            raise ValueError("code_challenge is required for PKCE flow")
        query.update({"code_challenge": code_challenge, "code_challenge_method": "S256"})

    return {"flow": flow, "auth_url": f"{XERO_AUTHORIZE_URL}?{urlencode(query)}"}


def oauth_callback(code: str, state: str | None = None, code_verifier: str | None = None) -> Dict[str, str]:
    if not code:
        raise ValueError("missing auth code")
    return {
        "status": "connected",
        "code": code,
        "state": state or "",
        "code_verifier": code_verifier or "",
    }
