"""Integration tests for the FastAPI endpoints."""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "x402_enabled" in data
    assert "providers_available" in data


# ---------------------------------------------------------------------------
# /providers
# ---------------------------------------------------------------------------

def test_providers_returns_list():
    r = client.get("/providers")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_providers_schema():
    r = client.get("/providers")
    for p in r.json():
        assert "id" in p
        assert "name" in p
        assert "model" in p
        assert "price_per_1k_tokens" in p
        assert "available" in p
        assert "supports_x402" in p


def test_providers_includes_ollama():
    r = client.get("/providers")
    ids = [p["id"] for p in r.json()]
    assert "ollama" in ids


def test_ollama_always_available():
    r = client.get("/providers")
    ollama = next(p for p in r.json() if p["id"] == "ollama")
    assert ollama["available"] is True


# ---------------------------------------------------------------------------
# /infer
# ---------------------------------------------------------------------------

def _mock_graph_result(provider_id="ollama", model="llama3.2"):
    from kite_utils import get_provider
    provider = get_provider(provider_id)
    return {
        "task": "Say hello",
        "task_type": "simple",
        "providers": [provider],
        "selected_provider": provider,
        "payment_receipt": {"status": "paid", "tx_hash": "mock_tx_abc12345", "amount_usd": 0.0, "provider": provider_id, "settled_on": "kite-testnet"},
        "response": "Hello, world!",
        "tokens_used": 42,
        "cost_usd": 0.0,
        "task_type": "simple",
    }


def test_infer_success():
    with patch("main.routing_graph.invoke", return_value=_mock_graph_result()):
        r = client.post("/infer", json={"prompt": "Say hello"})
    assert r.status_code == 200
    data = r.json()
    assert data["response"] == "Hello, world!"
    assert data["provider"] == "ollama"
    assert data["tokens_used"] == 42
    assert "tx_hash" in data


def test_infer_response_schema():
    with patch("main.routing_graph.invoke", return_value=_mock_graph_result()):
        r = client.post("/infer", json={"prompt": "test"})
    data = r.json()
    for field in ("response", "provider", "model", "tokens_used", "cost_usd", "tx_hash", "task_type"):
        assert field in data, f"Missing field: {field}"


def test_infer_empty_prompt_rejected():
    r = client.post("/infer", json={"prompt": ""})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_infer_whitespace_prompt_rejected():
    r = client.post("/infer", json={"prompt": "   "})
    assert r.status_code == 400


def test_infer_missing_prompt_field():
    r = client.post("/infer", json={})
    assert r.status_code == 422  # Pydantic validation error


def test_cors_headers_present():
    r = client.options(
        "/infer",
        headers={
            "Origin": "http://localhost",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"


def test_infer_routes_to_cheapest_for_simple_task():
    """Verify simple tasks are routed to the cheapest available provider."""
    with patch("main.routing_graph.invoke", return_value=_mock_graph_result("ollama")) as mock_invoke:
        client.post("/infer", json={"prompt": "Hi"})
        call_args = mock_invoke.call_args[0][0]
        assert call_args["task"] == "Hi"


# ---------------------------------------------------------------------------
# /pay
# ---------------------------------------------------------------------------

def test_pay_returns_token():
    r = client.post("/pay", json={"nonce": "abc123", "wallet_address": "0xDEADBEEF"})
    assert r.status_code == 200
    assert "payment_token" in r.json()
    assert len(r.json()["payment_token"]) > 10


def test_pay_missing_fields():
    r = client.post("/pay", json={})
    assert r.status_code == 422


def test_pay_empty_nonce():
    r = client.post("/pay", json={"nonce": "", "wallet_address": "0xDEAD"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# X402 flow
# ---------------------------------------------------------------------------

def test_x402_disabled_by_default_no_payment_needed():
    """With X402 off (default), /infer works without a payment token."""
    with patch("main.routing_graph.invoke", return_value=_mock_graph_result()):
        r = client.post("/infer", json={"prompt": "hello"})
    assert r.status_code == 200


def test_x402_returns_402_when_enabled():
    """With X402 on, /infer without a token should return 402."""
    with patch.dict(os.environ, {"KITE_X402_ENABLED": "true"}):
        with patch("main.x402_enabled", return_value=True):
            r = client.post("/infer", json={"prompt": "hello"})
    assert r.status_code == 402


def test_x402_full_flow():
    """Pay → get token → infer succeeds."""
    with patch("main.x402_enabled", return_value=True):
        # Step 1: hit /infer without token → 402 with challenge
        r1 = client.post("/infer", json={"prompt": "hello"})
        assert r1.status_code == 402
        challenge = r1.json()["detail"]["challenge"]

        # Step 2: pay with nonce
        r2 = client.post("/pay", json={
            "nonce": challenge["nonce"],
            "wallet_address": "0xTEST",
        })
        assert r2.status_code == 200
        token = r2.json()["payment_token"]

        # Step 3: retry /infer with token
        with patch("main.routing_graph.invoke", return_value=_mock_graph_result()):
            r3 = client.post(
                "/infer",
                json={"prompt": "hello"},
                headers={"X-Payment-Token": token},
            )
        assert r3.status_code == 200
        assert r3.json()["response"] == "Hello, world!"


def test_x402_token_single_use():
    """A payment token can only be used once."""
    with patch("main.x402_enabled", return_value=True):
        r_pay = client.post("/pay", json={"nonce": "n1", "wallet_address": "0xA"})
        token = r_pay.json()["payment_token"]

        with patch("main.routing_graph.invoke", return_value=_mock_graph_result()):
            r1 = client.post("/infer", json={"prompt": "hi"}, headers={"X-Payment-Token": token})
        assert r1.status_code == 200

        # Second use of the same token should fail
        r2 = client.post("/infer", json={"prompt": "hi"}, headers={"X-Payment-Token": token})
        assert r2.status_code == 402


# ---------------------------------------------------------------------------
# /infer/stream
# ---------------------------------------------------------------------------

async def _fake_stream(provider, prompt):
    for chunk in ["Hello", ", ", "world", "!"]:
        yield chunk


def test_stream_endpoint_returns_sse():
    with patch("main.stream_provider", side_effect=_fake_stream):
        r = client.post("/infer/stream", json={"prompt": "hi"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]


def test_stream_contains_meta_and_done():
    with patch("main.stream_provider", side_effect=_fake_stream):
        r = client.post("/infer/stream", json={"prompt": "hi"})
    body = r.text
    assert '"type": "meta"' in body
    assert '"type": "done"' in body


def test_stream_contains_chunks():
    with patch("main.stream_provider", side_effect=_fake_stream):
        r = client.post("/infer/stream", json={"prompt": "hi"})
    body = r.text
    assert '"type": "chunk"' in body
    assert "Hello" in body


def test_stream_empty_prompt_rejected():
    r = client.post("/infer/stream", json={"prompt": ""})
    assert r.status_code == 400
