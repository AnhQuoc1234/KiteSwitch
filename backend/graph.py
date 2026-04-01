from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from kite_utils import (
    Provider,
    get_available_providers,
    score_providers,
    pay_provider,
    invoke_provider,
)


class GraphState(TypedDict):
    task: str
    providers: list[Provider]
    selected_provider: Optional[Provider]
    payment_receipt: Optional[dict]
    response: Optional[str]
    tokens_used: Optional[int]
    cost_usd: Optional[float]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify_task(state: GraphState) -> GraphState:
    """
    Placeholder for task classification.
    In the cost-only phase this is a passthrough.
    Future: detect complexity, code vs. chat, etc.
    """
    return {**state, "providers": get_available_providers()}


def score_providers_node(state: GraphState) -> GraphState:
    """Sort available providers cheapest-first and pick the best one."""
    ranked = score_providers(state["providers"])
    if not ranked:
        raise RuntimeError("No available providers — check your API keys.")
    return {**state, "providers": ranked, "selected_provider": ranked[0]}


def pay_provider_node(state: GraphState) -> GraphState:
    """Simulate X402 payment for the selected provider."""
    provider = state["selected_provider"]
    # Estimate cost for ~500 tokens (unknown at pay-time; settle actual after inference)
    estimated_cost = provider.price_per_1k_tokens * 0.5
    receipt = pay_provider(provider, estimated_cost)
    return {**state, "payment_receipt": receipt}


def call_provider_node(state: GraphState) -> GraphState:
    """Run LLM inference and record actual token usage + cost."""
    provider = state["selected_provider"]
    response, tokens = invoke_provider(provider, state["task"])
    actual_cost = provider.price_per_1k_tokens * (tokens / 1000)
    return {
        **state,
        "response": response,
        "tokens_used": tokens,
        "cost_usd": round(actual_cost, 8),
    }


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    g.add_node("classify_task", classify_task)
    g.add_node("score_providers", score_providers_node)
    g.add_node("pay_provider", pay_provider_node)
    g.add_node("call_provider", call_provider_node)

    g.add_edge(START, "classify_task")
    g.add_edge("classify_task", "score_providers")
    g.add_edge("score_providers", "pay_provider")
    g.add_edge("pay_provider", "call_provider")
    g.add_edge("call_provider", END)

    return g.compile()


routing_graph = build_graph()
