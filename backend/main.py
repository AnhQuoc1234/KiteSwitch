import asyncio
import json
import os
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from graph import routing_graph
from kite_utils import (
    get_available_providers,
    PROVIDER_REGISTRY,
    classify_task,
    score_providers,
    pay_provider,
    stream_provider,
)
from x402 import (
    x402_enabled,
    create_payment_challenge,
    issue_payment_token,
    consume_payment_token,
)

app = FastAPI(title="KiteSwitch", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Payment-Required"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    prompt: str


class InferResponse(BaseModel):
    response: str
    provider: str
    model: str
    tokens_used: int
    cost_usd: float
    tx_hash: str
    task_type: str


class PayRequest(BaseModel):
    nonce: str
    wallet_address: str


class PayResponse(BaseModel):
    payment_token: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_x402(request: Request, provider_id: str, estimated_cost: float) -> None:
    """
    If X402 is enabled, verify that the request carries a valid payment token.
    Raises HTTPException(402) with a payment challenge if not.
    """
    if not x402_enabled():
        return

    token = request.headers.get("X-Payment-Token")
    if not token:
        challenge = create_payment_challenge(provider_id, estimated_cost)
        raise HTTPException(
            status_code=402,
            detail={
                "error": "Payment required",
                "challenge": challenge,
            },
            headers={"X-Payment-Required": json.dumps(challenge)},
        )

    receipt = consume_payment_token(token)
    if receipt is None:
        raise HTTPException(status_code=402, detail="Invalid or expired payment token.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest, request: Request):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    # Quick pre-flight to determine provider + estimated cost for X402 check
    available = get_available_providers()
    if not available:
        raise HTTPException(status_code=503, detail="No providers available.")

    task_type = classify_task(req.prompt)
    ranked = score_providers(available, task_type)
    estimated_cost = ranked[0].price_per_1k_tokens * 0.5

    _check_x402(request, ranked[0].id, estimated_cost)

    initial_state = {
        "task": req.prompt,
        "task_type": None,
        "providers": [],
        "selected_provider": None,
        "payment_receipt": None,
        "response": None,
        "tokens_used": None,
        "cost_usd": None,
    }

    # Run the synchronous graph in a thread so the event loop stays free
    result = await asyncio.to_thread(routing_graph.invoke, initial_state)

    provider = result["selected_provider"]
    return InferResponse(
        response=result["response"],
        provider=provider.id,
        model=provider.model,
        tokens_used=result["tokens_used"],
        cost_usd=result["cost_usd"],
        tx_hash=result["payment_receipt"]["tx_hash"],
        task_type=result["task_type"],
    )


@app.post("/infer/stream")
async def infer_stream(req: InferRequest, request: Request):
    """
    SSE streaming endpoint.

    Emits:
      data: {"type": "meta", "provider": "...", "model": "...", "task_type": "..."}
      data: {"type": "chunk", "text": "..."}   (one per token chunk)
      data: {"type": "done", "cost_usd": ..., "tx_hash": "..."}
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    available = get_available_providers()
    if not available:
        raise HTTPException(status_code=503, detail="No providers available.")

    task_type = classify_task(req.prompt)
    ranked = score_providers(available, task_type)
    estimated_cost = ranked[0].price_per_1k_tokens * 0.5

    _check_x402(request, ranked[0].id, estimated_cost)

    receipt = pay_provider(ranked[0], estimated_cost)

    async def event_stream() -> AsyncGenerator[str, None]:
        last_error = None

        for provider in ranked:
            try:
                meta = json.dumps({
                    "type": "meta",
                    "provider": provider.id,
                    "model": provider.model,
                    "task_type": task_type,
                })
                yield f"data: {meta}\n\n"

                total_chunks = 0
                async for chunk in stream_provider(provider, req.prompt):
                    payload = json.dumps({"type": "chunk", "text": chunk})
                    yield f"data: {payload}\n\n"
                    total_chunks += 1

                done = json.dumps({
                    "type": "done",
                    "cost_usd": provider.price_per_1k_tokens * total_chunks * 0.004,
                    "tx_hash": receipt["tx_hash"],
                })
                yield f"data: {done}\n\n"
                return

            except Exception as exc:
                last_error = exc
                err = json.dumps({"type": "error", "provider": provider.id, "detail": str(exc)})
                yield f"data: {err}\n\n"
                continue

        fatal = json.dumps({"type": "fatal", "detail": f"All providers failed: {last_error}"})
        yield f"data: {fatal}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/pay", response_model=PayResponse)
async def pay(req: PayRequest):
    """
    Simulate Kite L1 payment verification and issue a single-use payment token.
    In production this would verify the on-chain transaction before issuing.
    """
    if not req.nonce or not req.wallet_address:
        raise HTTPException(status_code=400, detail="nonce and wallet_address are required.")
    token = issue_payment_token(req.nonce, req.wallet_address)
    return PayResponse(payment_token=token)


@app.get("/providers")
async def list_providers():
    available_ids = {p.id for p in get_available_providers()}
    return [
        {
            "id": p.id,
            "name": p.name,
            "model": p.model,
            "price_per_1k_tokens": p.price_per_1k_tokens,
            "available": p.id in available_ids,
            "supports_x402": p.supports_x402,
            "quality_score": p.quality_score,
        }
        for p in PROVIDER_REGISTRY
    ]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "x402_enabled": x402_enabled(),
        "providers_available": len(get_available_providers()),
    }
