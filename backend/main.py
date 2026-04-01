from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from graph import routing_graph
from kite_utils import get_available_providers, PROVIDER_REGISTRY

app = FastAPI(title="KiteSwitch", version="0.1.0")


class InferRequest(BaseModel):
    prompt: str


class InferResponse(BaseModel):
    response: str
    provider: str
    model: str
    tokens_used: int
    cost_usd: float
    tx_hash: str


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    initial_state = {
        "task": req.prompt,
        "providers": [],
        "selected_provider": None,
        "payment_receipt": None,
        "response": None,
        "tokens_used": None,
        "cost_usd": None,
    }

    result = routing_graph.invoke(initial_state)

    provider = result["selected_provider"]
    return InferResponse(
        response=result["response"],
        provider=provider.id,
        model=provider.model,
        tokens_used=result["tokens_used"],
        cost_usd=result["cost_usd"],
        tx_hash=result["payment_receipt"]["tx_hash"],
    )


@app.get("/providers")
def list_providers():
    available_ids = {p.id for p in get_available_providers()}
    return [
        {
            "id": p.id,
            "name": p.name,
            "model": p.model,
            "price_per_1k_tokens": p.price_per_1k_tokens,
            "available": p.id in available_ids,
            "supports_x402": p.supports_x402,
        }
        for p in PROVIDER_REGISTRY
    ]


@app.get("/health")
def health():
    return {"status": "ok"}
