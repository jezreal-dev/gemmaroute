"""
routers/route_endpoint.py — POST /route

The primary API endpoint. Accepts a prompt, builds the initial AgentState,
invokes the LangGraph routing pipeline asynchronously, and returns the
structured RouteResponse including full routing metadata.
"""
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

from graph.builder import routing_graph
from graph.state import AgentState

router = APIRouter(tags=["routing"])


# ── Request / Response models ─────────────────────────────────────────────────

class RouteRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096, description="Customer support query")
    session_id: Optional[str] = Field("default", description="Optional session identifier")
    max_cost_tier: Optional[str] = Field(
        "complex",
        description="Cap routing at this tier: 'simple' | 'medium' | 'complex'",
    )


class RoutingMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    initial_tier: Optional[str]
    final_tier: Optional[str]
    escalations: int
    classifier_confidence: float
    quality_score: float
    model_used: Optional[str]
    latency_ms: float
    estimated_cost_usd: float
    cost_saved_vs_max_usd: float


class RouteResponse(BaseModel):
    response: str
    routing: RoutingMetadata
    session_id: str


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/route",
    response_model=RouteResponse,
    summary="Route a prompt through the 3-layer GemmaRoute engine",
    description=(
        "Classifies the prompt using a local Gemma 4 model, routes to the "
        "cheapest capable tier (local/medium cloud/complex cloud), validates "
        "output quality with an LLM-as-judge, and returns the response with "
        "full observability metadata."
    ),
)
async def route_prompt(request: RouteRequest):
    initial_state: AgentState = {
        # Input
        "prompt":        request.prompt,
        "session_id":    request.session_id or "default",
        "max_cost_tier": request.max_cost_tier or "complex",
        # Filter (defaults)
        "is_trivial":      False,
        "trivial_response": None,
        # Classifier (defaults)
        "initial_tier":          None,
        "current_tier":          None,
        "classifier_confidence": 0.0,
        # Executor (defaults)
        "response":    None,
        "model_used":  None,
        "tokens_used": 0,
        # Judge (defaults)
        "quality_score":         0.0,
        "escalation_depth":      0,
        "hop_budget_exhausted":  False,
        # Observability
        "start_time": time.time(),
        "latency_ms": 0.0,
        "cost_usd":   0.0,
        "saved_usd":  0.0,
    }

    try:
        final_state = await routing_graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Routing pipeline error: {exc}",
        ) from exc

    return RouteResponse(
        response=final_state.get("response") or "",
        routing=RoutingMetadata(
            initial_tier          = final_state.get("initial_tier"),
            final_tier            = final_state.get("current_tier"),
            escalations           = final_state.get("escalation_depth", 0),
            classifier_confidence = final_state.get("classifier_confidence", 0.0),
            quality_score         = final_state.get("quality_score", 0.0),
            model_used            = final_state.get("model_used"),
            latency_ms            = final_state.get("latency_ms", 0.0),
            estimated_cost_usd    = final_state.get("cost_usd", 0.0),
            cost_saved_vs_max_usd = final_state.get("saved_usd", 0.0),
        ),
        session_id=final_state.get("session_id", ""),
    )
