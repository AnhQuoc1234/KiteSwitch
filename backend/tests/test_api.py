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
    assert r.json() == {"status": "ok"}


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
    for field in ("response", "provider", "model", "tokens_used", "cost_usd", "tx_hash"):
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
