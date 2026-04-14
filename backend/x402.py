"""
X402 payment protocol helpers.

Flow:
  1. Client POST /infer  →  402 + X-Payment-Required header (JSON challenge)
  2. Client POST /pay    →  200 + {"payment_token": "..."}
  3. Client POST /infer with X-Payment-Token header  →  200 LLM response

Enable with KITE_X402_ENABLED=true in .env.
Default is disabled so the API stays open during development.
"""
import os
import time
import secrets
from typing import Optional


def x402_enabled() -> bool:
    return os.getenv("KITE_X402_ENABLED", "false").lower() == "true"


# ---------------------------------------------------------------------------
# In-memory token store (swap for Redis in production)
# ---------------------------------------------------------------------------

_issued_tokens: dict[str, dict] = {}
_TOKEN_TTL = 300  # 5 minutes


def create_payment_challenge(provider_id: str, estimated_cost_usd: float) -> dict:
    """
    Return a payment challenge dict that goes in the X-Payment-Required header.
    The nonce ties the challenge to the eventual payment token.
    """
    return {
        "nonce": secrets.token_hex(16),
        "amount_usd": round(estimated_cost_usd, 8),
        "currency": "KITE",
        "network": "kite-testnet",
        "provider": provider_id,
        "payment_url": "/pay",
        "expires_at": int(time.time()) + _TOKEN_TTL,
    }


def issue_payment_token(nonce: str, wallet_address: str) -> str:
    """
    Simulate receiving on-chain payment proof and issue a short-lived token.
    In production this would verify the on-chain tx before issuing.
    """
    token = secrets.token_urlsafe(32)
    _issued_tokens[token] = {
        "nonce": nonce,
        "wallet": wallet_address,
        "issued_at": time.time(),
        "expires_at": time.time() + _TOKEN_TTL,
    }
    return token


def consume_payment_token(token: str) -> Optional[dict]:
    """
    Verify and consume (single-use) a payment token.
    Returns the token metadata on success, None if invalid/expired.
    """
    info = _issued_tokens.get(token)
    if not info:
        return None
    if time.time() > info["expires_at"]:
        _issued_tokens.pop(token, None)
        return None
    return _issued_tokens.pop(token)
