from __future__ import annotations

from dataclasses import dataclass
import os


DEFAULT_XERO_SCOPES = (
    "offline_access",
    "accounting.transactions",
    "accounting.contacts",
    "accounting.settings",
    "files",
)


@dataclass
class Settings:
    firebase_project_id: str = os.getenv("FIREBASE_PROJECT_ID", "")
    xero_client_id: str = os.getenv("XERO_CLIENT_ID", "")
    xero_client_secret: str = os.getenv("XERO_CLIENT_SECRET", "")
    xero_redirect_uri: str = os.getenv("XERO_REDIRECT_URI", "")
    xero_scopes: tuple[str, ...] = tuple(
        scope.strip()
        for scope in os.getenv("XERO_SCOPES", " ".join(DEFAULT_XERO_SCOPES)).split()
        if scope.strip()
    )


settings = Settings()
