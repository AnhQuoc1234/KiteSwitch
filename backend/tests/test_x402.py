"""Tests for the X402 payment protocol module."""
import os
import sys
import time
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from x402 import (
    x402_enabled,
    create_payment_challenge,
    issue_payment_token,
    consume_payment_token,
    _issued_tokens,
)


# ---------------------------------------------------------------------------
# x402_enabled
# ---------------------------------------------------------------------------

def test_x402_disabled_by_default():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("KITE_X402_ENABLED", None)
        assert x402_enabled() is False


def test_x402_enabled_when_set():
    with patch.dict(os.environ, {"KITE_X402_ENABLED": "true"}):
        assert x402_enabled() is True


def test_x402_case_insensitive():
    with patch.dict(os.environ, {"KITE_X402_ENABLED": "TRUE"}):
        assert x402_enabled() is True


# ---------------------------------------------------------------------------
# create_payment_challenge
# ---------------------------------------------------------------------------

def test_challenge_has_required_fields():
    c = create_payment_challenge("openai", 0.005)
    for field in ("nonce", "amount_usd", "currency", "network", "provider", "payment_url", "expires_at"):
        assert field in c, f"Missing field: {field}"


def test_challenge_amount_rounded():
    c = create_payment_challenge("openai", 0.123456789)
    assert c["amount_usd"] == round(0.123456789, 8)


def test_challenge_provider_set():
    c = create_payment_challenge("anthropic", 0.001)
    assert c["provider"] == "anthropic"


def test_challenge_nonce_unique():
    c1 = create_payment_challenge("openai", 0.001)
    c2 = create_payment_challenge("openai", 0.001)
    assert c1["nonce"] != c2["nonce"]


def test_challenge_expires_in_future():
    c = create_payment_challenge("openai", 0.001)
    assert c["expires_at"] > int(time.time())


def test_challenge_payment_url():
    c = create_payment_challenge("openai", 0.001)
    assert c["payment_url"] == "/pay"


# ---------------------------------------------------------------------------
# issue_payment_token / consume_payment_token
# ---------------------------------------------------------------------------

def test_issue_returns_token_string():
    token = issue_payment_token("nonce123", "0xWALLET")
    assert isinstance(token, str)
    assert len(token) > 10


def test_consume_valid_token():
    token = issue_payment_token("nonce_abc", "0xWALLET")
    result = consume_payment_token(token)
    assert result is not None
    assert result["nonce"] == "nonce_abc"
    assert result["wallet"] == "0xWALLET"


def test_consume_single_use():
    token = issue_payment_token("nonce_su", "0xWALLET")
    assert consume_payment_token(token) is not None
    assert consume_payment_token(token) is None  # second use fails


def test_consume_unknown_token():
    assert consume_payment_token("not_a_real_token") is None


def test_consume_expired_token():
    token = issue_payment_token("nonce_exp", "0xWALLET")
    # Manually expire the token
    _issued_tokens[token]["expires_at"] = time.time() - 1
    assert consume_payment_token(token) is None


def test_unique_tokens_per_issue():
    t1 = issue_payment_token("n1", "0xA")
    t2 = issue_payment_token("n2", "0xB")
    assert t1 != t2
