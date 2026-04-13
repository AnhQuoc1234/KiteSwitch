"""Unit tests for kite_utils: Provider, classify_task, score_providers."""
import os
import pytest
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from kite_utils import (
    Provider,
    PROVIDER_REGISTRY,
    classify_task,
    get_available_providers,
    get_provider,
    score_providers,
    pay_provider,
)


# ---------------------------------------------------------------------------
# Provider.available
# ---------------------------------------------------------------------------

class TestProviderAvailable:
    def test_no_key_env_always_available(self):
        p = Provider(id="local", name="Local", model="x", price_per_1k_tokens=0.0, base_url="http://x")
        assert p.available is True

    def test_key_env_set(self):
        p = Provider(id="x", name="X", model="x", price_per_1k_tokens=0.0, base_url="http://x", api_key_env="TEST_KEY_XYZ")
        with patch.dict(os.environ, {"TEST_KEY_XYZ": "sk-test"}):
            assert p.available is True

    def test_key_env_missing(self):
        p = Provider(id="x", name="X", model="x", price_per_1k_tokens=0.0, base_url="http://x", api_key_env="MISSING_KEY_XYZ")
        os.environ.pop("MISSING_KEY_XYZ", None)
        assert p.available is False

    def test_api_key_property(self):
        p = Provider(id="x", name="X", model="x", price_per_1k_tokens=0.0, base_url="http://x", api_key_env="MY_KEY")
        with patch.dict(os.environ, {"MY_KEY": "secret"}):
            assert p.api_key == "secret"

    def test_api_key_none_when_no_env(self):
        p = Provider(id="x", name="X", model="x", price_per_1k_tokens=0.0, base_url="http://x")
        assert p.api_key is None


# ---------------------------------------------------------------------------
# classify_task
# ---------------------------------------------------------------------------

class TestClassifyTask:
    def test_code_keyword(self):
        assert classify_task("Write a Python function to sort a list") == "code"

    def test_debug_keyword(self):
        assert classify_task("Debug this javascript error") == "code"

    def test_api_keyword(self):
        assert classify_task("Design an API endpoint for users") == "code"

    def test_complex_keyword(self):
        assert classify_task("Explain the strategy for microservice architecture") == "complex"

    def test_long_prompt_is_complex(self):
        # 9 words × 10 = 90 words, clearly > 80
        long_prompt = "What do you think about the state of the world? " * 10
        assert classify_task(long_prompt) == "complex"

    def test_simple_short_prompt(self):
        assert classify_task("What is 2 + 2?") == "simple"

    def test_simple_greeting(self):
        assert classify_task("Hello, how are you?") == "simple"

    def test_case_insensitive(self):
        assert classify_task("IMPLEMENT a solution") == "code"


# ---------------------------------------------------------------------------
# get_available_providers / get_provider
# ---------------------------------------------------------------------------

class TestGetProviders:
    def test_get_provider_by_id(self):
        p = get_provider("openai")
        assert p is not None
        assert p.id == "openai"

    def test_get_provider_missing(self):
        assert get_provider("nonexistent") is None

    def test_ollama_always_available(self):
        """Ollama has no api_key_env so it's always available."""
        p = get_provider("ollama")
        assert p.available is True

    def test_get_available_includes_ollama(self):
        available_ids = {p.id for p in get_available_providers()}
        assert "ollama" in available_ids

    def test_openai_unavailable_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove openai key if set
            os.environ.pop("OPENAI_API_KEY", None)
            p = get_provider("openai")
            assert p.available is False


# ---------------------------------------------------------------------------
# score_providers
# ---------------------------------------------------------------------------

def make_providers():
    return [
        Provider(id="cheap", name="Cheap", model="x", price_per_1k_tokens=0.0001, base_url="http://x", quality_score=1.0),
        Provider(id="mid",   name="Mid",   model="x", price_per_1k_tokens=0.001,  base_url="http://x", quality_score=1.5),
        Provider(id="best",  name="Best",  model="x", price_per_1k_tokens=0.01,   base_url="http://x", quality_score=2.0),
    ]


class TestScoreProviders:
    def test_simple_cheapest_first(self):
        providers = make_providers()
        ranked = score_providers(providers, "simple")
        assert ranked[0].id == "cheap"

    def test_complex_quality_first(self):
        providers = make_providers()
        ranked = score_providers(providers, "complex")
        assert ranked[0].id == "best"

    def test_code_balances_quality_and_cost(self):
        providers = make_providers()
        ranked = score_providers(providers, "code")
        # Should NOT be cheapest (quality matters), but also not purely quality-only
        assert ranked[0].id in ("mid", "best")

    def test_empty_providers(self):
        assert score_providers([], "simple") == []

    def test_single_provider(self):
        p = make_providers()[0]
        assert score_providers([p], "complex") == [p]

    def test_free_provider_wins_simple(self):
        providers = [
            Provider(id="free", name="Free", model="x", price_per_1k_tokens=0.0, base_url="http://x"),
            Provider(id="paid", name="Paid", model="x", price_per_1k_tokens=0.001, base_url="http://x"),
        ]
        ranked = score_providers(providers, "simple")
        assert ranked[0].id == "free"

    def test_unknown_task_type_falls_back_to_cheapest(self):
        providers = make_providers()
        ranked = score_providers(providers, "unknown_type")
        assert ranked[0].id == "cheap"


# ---------------------------------------------------------------------------
# pay_provider
# ---------------------------------------------------------------------------

class TestPayProvider:
    def test_returns_paid_status(self):
        p = make_providers()[0]
        receipt = pay_provider(p, 0.005)
        assert receipt["status"] == "paid"

    def test_tx_hash_format(self):
        p = make_providers()[0]
        receipt = pay_provider(p, 0.005)
        assert receipt["tx_hash"].startswith("mock_tx_")

    def test_amount_rounded(self):
        p = make_providers()[0]
        receipt = pay_provider(p, 0.123456789)
        assert receipt["amount_usd"] == round(0.123456789, 8)

    def test_provider_id_in_receipt(self):
        p = make_providers()[0]
        receipt = pay_provider(p, 0.001)
        assert receipt["provider"] == "cheap"

    def test_unique_tx_hashes(self):
        p = make_providers()[0]
        r1 = pay_provider(p, 0.001)
        r2 = pay_provider(p, 0.001)
        assert r1["tx_hash"] != r2["tx_hash"]
