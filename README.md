# KiteSwitch

Multi-agent routing layer that uses the X402 protocol to autonomously discover, pay for, and settle the most cost-effective LLM inference for any given task on the Kite L1 blockchain.

## Architecture

```
POST /infer
     │
     ▼
[classify_task]     ← passthrough (future: complexity detection)
     │
     ▼
[score_providers]   ← sort available providers by price_per_1k_tokens
     │
     ▼
[pay_provider]      ← mock X402 payment (future: real Kite L1 settlement)
     │
     ▼
[call_provider]     ← LLM inference (OpenAI / Anthropic / Ollama)
     │
     ▼
InferResponse
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn langgraph langchain-openai anthropic python-dotenv openai
```

Copy `.env_sample` to `.env` and fill in your keys:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
KITE_AGENT_ID=...
KITE_WALLET_ADDRESS=...
```

Ollama (optional, free): install from [ollama.com](https://ollama.com) and run `ollama pull llama3.2`.

## Run

```bash
cd backend
uvicorn main:app --reload
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/infer` | Route a prompt to the cheapest available provider |
| GET | `/providers` | List all providers and their availability |
| GET | `/health` | Health check |

## Provider priority (cost-only)

| Provider | Model | Price/1k tokens |
|----------|-------|----------------|
| Ollama (local) | llama3.2 | $0.00 |
| OpenAI | gpt-4o-mini | $0.00015 |
| Anthropic | claude-haiku-4-5 | $0.00025 |
