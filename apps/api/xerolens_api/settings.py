from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class Settings:
    firebase_project_id: str = os.getenv("FIREBASE_PROJECT_ID", "")
    xero_client_id: str = os.getenv("XERO_CLIENT_ID", "")
    xero_client_secret: str = os.getenv("XERO_CLIENT_SECRET", "")
    xero_redirect_uri: str = os.getenv("XERO_REDIRECT_URI", "")


settings = Settings()
