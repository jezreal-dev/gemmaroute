"""
graph/state.py — AgentState TypedDict for the LangGraph routing pipeline.

Every field is populated incrementally as the request flows through the graph:
  START → filter → [trivial_respond | classifier] → executor → judge → logger → END

LangGraph merges partial dicts returned by each node into the full state,
so nodes only need to return the fields they modify.
"""
from typing import TypedDict, Optional


class AgentState(TypedDict):
    # ── Input (set at graph entry in route_endpoint.py) ────────────────────
    prompt: str
    session_id: str
    max_cost_tier: str          # "simple" | "medium" | "complex" — user-defined cap

    # ── Heuristic filter (set by heuristic_filter_node) ────────────────────
    is_trivial: bool
    trivial_response: Optional[str]

    # ── Classifier output (set by gemma_classifier_node) ───────────────────
    initial_tier: Optional[str]     # First classification (never changes)
    current_tier: Optional[str]     # Active tier (changes on escalation)
    classifier_confidence: float    # Gemma's confidence in its classification

    # ── Executor output (set by executor nodes) ─────────────────────────────
    response: Optional[str]         # Generated text response
    model_used: Optional[str]       # Exact model ID that produced the response
    tokens_used: int                # Total tokens consumed

    # ── Quality judge (set by quality_judge_node) ───────────────────────────
    quality_score: float            # LLM-as-judge score: 0.0–1.0
    escalation_depth: int           # Times we've upgraded the tier

    # ── Observability (set progressively) ───────────────────────────────────
    start_time: float               # Unix timestamp at graph entry
    latency_ms: float               # Final end-to-end latency (set by logger)
    cost_usd: float                 # Actual API cost (set by logger)
    saved_usd: float                # Cost saved vs always-complex (set by logger)
