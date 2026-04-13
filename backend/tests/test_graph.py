"""Unit tests for graph nodes (no real LLM calls)."""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from kite_utils import Provider
from graph import (
    GraphState,
    classify_task_node,
    score_providers_node,
    pay_provider_node,
    call_provider_node,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_state(**kwargs) -> GraphState:
    defaults: GraphState = {
        "task": "Hello world",
        "task_type": None,
        "providers": [],
        "selected_provider": None,
        "payment_receipt": None,
        "response": None,
        "tokens_used": None,
        "cost_usd": None,
    }
    defaults.update(kwargs)
    return defaults


def fake_provider(**kwargs) -> Provider:
    defaults = dict(id="test", name="Test", model="test-model",
                    price_per_1k_tokens=0.001, base_url="http://test", quality_score=1.0)
    defaults.update(kwargs)
    return Provider(**defaults)


# ---------------------------------------------------------------------------
# classify_task_node
# ---------------------------------------------------------------------------

class TestClassifyTaskNode:
    def test_sets_task_type_simple(self):
        state = base_state(task="What is 2+2?")
        with patch("graph.get_available_providers", return_value=[]):
            result = classify_task_node(state)
        assert result["task_type"] == "simple"

    def test_sets_task_type_code(self):
        state = base_state(task="Write a Python function to sort a list")
        with patch("graph.get_available_providers", return_value=[]):
            result = classify_task_node(state)
        assert result["task_type"] == "code"

    def test_populates_providers(self):
        p = fake_provider()
        state = base_state(task="hello")
        with patch("graph.get_available_providers", return_value=[p]):
            result = classify_task_node(state)
        assert result["providers"] == [p]


# ---------------------------------------------------------------------------
# score_providers_node
# ---------------------------------------------------------------------------

class TestScoreProvidersNode:
    def test_picks_best_provider(self):
        cheap = fake_provider(id="cheap", price_per_1k_tokens=0.001)
        expensive = fake_provider(id="expensive", price_per_1k_tokens=0.01)
        state = base_state(task="hello", task_type="simple", providers=[expensive, cheap])
        result = score_providers_node(state)
        assert result["selected_provider"].id == "cheap"

    def test_raises_when_no_providers(self):
        state = base_state(task="hello", task_type="simple", providers=[])
        with pytest.raises(RuntimeError, match="No available providers"):
            score_providers_node(state)

    def test_preserves_all_providers_ranked(self):
        providers = [fake_provider(id=str(i), price_per_1k_tokens=float(i)) for i in range(3, 0, -1)]
        state = base_state(task="hello", task_type="simple", providers=providers)
        result = score_providers_node(state)
        assert result["providers"][0].id == "1"


# ---------------------------------------------------------------------------
# pay_provider_node
# ---------------------------------------------------------------------------

class TestPayProviderNode:
    def test_receipt_attached(self):
        p = fake_provider()
        state = base_state(selected_provider=p)
        result = pay_provider_node(state)
        assert result["payment_receipt"]["status"] == "paid"
        assert result["payment_receipt"]["tx_hash"].startswith("mock_tx_")

    def test_estimated_cost_calculation(self):
        p = fake_provider(price_per_1k_tokens=0.002)
        state = base_state(selected_provider=p)
        result = pay_provider_node(state)
        assert result["payment_receipt"]["amount_usd"] == round(0.002 * 0.5, 8)


# ---------------------------------------------------------------------------
# call_provider_node
# ---------------------------------------------------------------------------

class TestCallProviderNode:
    def _state_with_provider(self, provider_id="openai"):
        p = fake_provider(id=provider_id, price_per_1k_tokens=0.001)
        return base_state(task="Say hello", selected_provider=p)

    def test_openai_provider_called(self):
        state = self._state_with_provider("openai")
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Hello!"
        mock_completion.usage.total_tokens = 20

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.return_value = mock_completion
            result = call_provider_node(state)

        assert result["response"] == "Hello!"
        assert result["tokens_used"] == 20
        assert result["cost_usd"] == round(0.001 * 20 / 1000, 8)

    def test_anthropic_provider_called(self):
        state = self._state_with_provider("anthropic")
        mock_msg = MagicMock()
        mock_msg.content[0].text = "Hi there!"
        mock_msg.usage.input_tokens = 10
        mock_msg.usage.output_tokens = 15

        with patch("anthropic.Anthropic") as MockAnthropicClass:
            MockAnthropicClass.return_value.messages.create.return_value = mock_msg
            result = call_provider_node(state)

        assert result["response"] == "Hi there!"
        assert result["tokens_used"] == 25

    def test_unknown_provider_raises(self):
        state = self._state_with_provider("unknown_provider")
        with pytest.raises(ValueError, match="Unknown provider"):
            call_provider_node(state)
