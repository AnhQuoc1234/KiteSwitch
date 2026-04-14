from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator
import os
import re


@dataclass
class Provider:
    id: str
    name: str
    model: str
    price_per_1k_tokens: float  # USD
    base_url: str
    api_key_env: Optional[str] = None   # env var holding the API key
    supports_x402: bool = False         # flip True once X402 discovery is wired
    quality_score: float = 1.0          # relative quality tier (higher = better)
    extra: dict = field(default_factory=dict)

    @property
    def api_key(self) -> Optional[str]:
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    @property
    def available(self) -> bool:
        """Provider is usable if it needs no key, or the key env var is set."""
        if self.api_key_env is None:
            return True
        return bool(self.api_key)


# ---------------------------------------------------------------------------
# Static registry — replace / extend with X402 discovery in a later phase
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: list[Provider] = [
    Provider(
        id="openai",
        name="OpenAI",
        model="gpt-4o-mini",
        price_per_1k_tokens=0.00015,
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        quality_score=1.5,
    ),
    Provider(
        id="anthropic",
        name="Anthropic",
        model="claude-haiku-4-5-20251001",
        price_per_1k_tokens=0.00025,
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        quality_score=1.7,
    ),
    Provider(
        id="ollama",
        name="Ollama (local)",
        model="llama3.2",
        price_per_1k_tokens=0.0,
        base_url="http://localhost:11434/v1",
        api_key_env=None,   # no key needed for local
        quality_score=1.0,
    ),
]

# ---------------------------------------------------------------------------
# Task classification
# ---------------------------------------------------------------------------

CODE_KEYWORDS = re.compile(
    r"\b(code|function|class|debug|implement|refactor|script|algorithm|"
    r"python|javascript|typescript|sql|api|bug|error|exception|compile|"
    r"test|unittest|async|regex|parse|json|xml|html|css)\b",
    re.IGNORECASE,
)

COMPLEX_KEYWORDS = re.compile(
    r"\b(explain|analyze|compare|summarize|research|design|architecture|"
    r"strategy|plan|review|evaluate|elaborate|detailed|comprehensive)\b",
    re.IGNORECASE,
)


def classify_task(prompt: str) -> str:
    """
    Classify a prompt into one of three task types:
      - "code"    → code generation / debugging / technical
      - "complex" → long-form reasoning, analysis, explanation
      - "simple"  → short / conversational / factual

    Returns one of: "code", "complex", "simple"
    """
    words = len(prompt.split())

    if CODE_KEYWORDS.search(prompt):
        return "code"
    if words > 80 or COMPLEX_KEYWORDS.search(prompt):
        return "complex"
    return "simple"


# ---------------------------------------------------------------------------
# Provider utilities
# ---------------------------------------------------------------------------

def get_available_providers() -> list[Provider]:
    """Return only providers whose credentials are present."""
    return [p for p in PROVIDER_REGISTRY if p.available]


def get_provider(provider_id: str) -> Optional[Provider]:
    for p in PROVIDER_REGISTRY:
        if p.id == provider_id:
            return p
    return None


def score_providers(providers: list[Provider], task_type: str = "simple") -> list[Provider]:
    """
    Score and rank providers.

    Scoring strategy by task type:
      - "simple"  → cheapest wins (pure cost sort)
      - "code"    → balance cost with quality (prefer capable models)
      - "complex" → quality-first (best model within available set)

    Returns providers sorted best-first.
    """
    if not providers:
        return []

    if task_type == "simple":
        return sorted(providers, key=lambda p: p.price_per_1k_tokens)

    if task_type == "code":
        max_cost = max(p.price_per_1k_tokens for p in providers) or 1e-9
        def code_score(p: Provider) -> float:
            cost_norm = p.price_per_1k_tokens / max_cost
            return -(0.6 * p.quality_score - 0.4 * cost_norm)
        return sorted(providers, key=code_score)

    if task_type == "complex":
        return sorted(providers, key=lambda p: (-p.quality_score, p.price_per_1k_tokens))

    return sorted(providers, key=lambda p: p.price_per_1k_tokens)


def pay_provider(provider: Provider, amount_usd: float) -> dict:
    """Simulate the X402 payment handshake. Swap for real Kite L1 settlement later."""
    from uuid import uuid4
    return {
        "status": "paid",
        "tx_hash": f"mock_tx_{uuid4().hex[:8]}",
        "amount_usd": round(amount_usd, 8),
        "provider": provider.id,
        "settled_on": "kite-testnet",
    }


def invoke_provider(provider: Provider, prompt: str) -> tuple[str, int]:
    """Synchronous LLM call. Returns (response_text, tokens_used)."""
    if provider.id in ("openai", "ollama"):
        from openai import OpenAI
        client = OpenAI(
            api_key=provider.api_key or "ollama",
            base_url=provider.base_url,
        )
        completion = client.chat.completions.create(
            model=provider.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content, completion.usage.total_tokens

    elif provider.id == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=provider.api_key)
        message = client.messages.create(
            model=provider.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        tokens = message.usage.input_tokens + message.usage.output_tokens
        return message.content[0].text, tokens

    raise ValueError(f"Unknown provider: {provider.id}")


async def invoke_provider_async(provider: Provider, prompt: str) -> tuple[str, int]:
    """Async LLM call. Returns (response_text, tokens_used)."""
    if provider.id in ("openai", "ollama"):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=provider.api_key or "ollama",
            base_url=provider.base_url,
        )
        completion = await client.chat.completions.create(
            model=provider.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content, completion.usage.total_tokens

    elif provider.id == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=provider.api_key)
        message = await client.messages.create(
            model=provider.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        tokens = message.usage.input_tokens + message.usage.output_tokens
        return message.content[0].text, tokens

    raise ValueError(f"Unknown provider: {provider.id}")


async def stream_provider(provider: Provider, prompt: str) -> AsyncGenerator[str, None]:
    """Yield text chunks from the provider via streaming."""
    if provider.id in ("openai", "ollama"):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=provider.api_key or "ollama",
            base_url=provider.base_url,
        )
        async with client.chat.completions.stream(
            model=provider.model,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    elif provider.id == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=provider.api_key)
        async with client.messages.stream(
            model=provider.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    else:
        raise ValueError(f"Unknown provider: {provider.id}")
