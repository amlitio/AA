from __future__ import annotations


def verify_firebase_token(id_token: str) -> dict:
    """Placeholder verification hook; wire to firebase_admin in production."""
    if not id_token:
        raise ValueError("Missing Firebase ID token")
    return {"uid": "mock-user", "token": id_token}
