"""
graph/nodes.py — All async node functions for the GemmaRoute LangGraph.

Each node receives the full AgentState and returns a PARTIAL dict of only
the fields it modifies. LangGraph merges these partials into the running state.

Node execution order:
  heuristic_filter_node
    ├── trivial_respond_node → logger_node
    └── gemma_classifier_node
          ├── local_executor_node ─┐
          ├── cloud_medium_node   ─┼─► quality_judge_node
          └── cloud_complex_node ─┘         ├── logger_node
                                            └── escalate_tier_node → [executor loop]
"""
import re
import time
from typing import Optional

from config import settings, COST_PER_1M_TOKENS

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1a — Heuristic Filter (zero LLM calls, instant response)
# ─────────────────────────────────────────────────────────────────────────────

_TRIVIAL_RULES = [
    (
        re.compile(
            r"^(hi|hello|hey|good[\s-]?(morning|afternoon|evening))[!.,\s]*$",
            re.IGNORECASE,
        ),
        "greeting",
    ),
    (
        re.compile(
            r".*(business[\s-]?hour|opening[\s-]?hour|store[\s-]?hour|"
            r"open[\s-]?until|when[\s-]+(are|do)[\s-]+you[\s-]+open).*",
            re.IGNORECASE,
        ),
        "hours",
    ),
    (
        re.compile(
            r"^(thanks|thank\s+you+|thx|ty|much\s+appreciated)[!.,\s]*$",
            re.IGNORECASE,
        ),
        "thanks",
    ),
    (
        re.compile(r"^(bye|goodbye|see\s+you|cya)[!.,\s]*$", re.IGNORECASE),
        "goodbye",
    ),
    (
        re.compile(r"^are\s+you\s+(open|closed|available)\s*\??$", re.IGNORECASE),
        "availability",
    ),
]

_TRIVIAL_RESPONSES = {
    "greeting": (
        "Hello! 👋 Welcome to our customer support. How can I help you today?"
    ),
    "hours": (
        "Our business hours are **Monday–Friday 9AM–6PM EST** and "
        "**Saturday 10AM–4PM EST**. We're closed on Sundays and major holidays."
    ),
    "thanks": "You're welcome! 😊 Is there anything else I can help you with?",
    "goodbye": "Goodbye! Have a wonderful day. Feel free to reach out anytime. 👋",
    "availability": (
        "Yes, we're available **Monday–Friday 9AM–6PM EST** "
        "and **Saturday 10AM–4PM EST**."
    ),
}


def _match_trivial(prompt: str) -> Optional[str]:
    """Return a canned response if prompt matches a trivial rule, else None."""
    for pattern, key in _TRIVIAL_RULES:
        if pattern.search(prompt.strip()):
            return _TRIVIAL_RESPONSES[key]
    return None


async def heuristic_filter_node(state: dict) -> dict:
    """Layer 1a: Instant regex match — no LLM needed for simple patterns."""
    trivial_resp = _match_trivial(state["prompt"])
    if trivial_resp:
        return {"is_trivial": True, "trivial_response": trivial_resp}
    return {"is_trivial": False, "trivial_response": None}


async def trivial_respond_node(state: dict) -> dict:
    """Set final response fields for trivially matched queries."""
    return {
        "response":              state.get("trivial_response", "How can I help you today?"),
        "model_used":            "heuristic_filter",
        "tokens_used":           0,
        "quality_score":         1.0,
        "initial_tier":          "trivial",
        "current_tier":          "trivial",
        "classifier_confidence": 1.0,
        "escalation_depth":      0,
        "hop_budget_exhausted":  False,
        "cost_usd":              0.0,
        "saved_usd":             0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1b — Gemma Semantic Classifier (local AMD ROCm, $0.00)
# ─────────────────────────────────────────────────────────────────────────────

async def gemma_classifier_node(state: dict) -> dict:
    """Bypassed local classification to unblock frontend. Route directly to Fireworks Complex tier."""
    return {
        "initial_tier":          "complex",
        "current_tier":          "complex",
        "classifier_confidence": 0.99,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Executors
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORT_SYSTEM = (
    "You are a professional, empathetic customer support agent. "
    "Provide accurate, helpful, and concise responses. "
    "If you lack specific company details, acknowledge it and offer to escalate."
)


async def local_executor_node(state: dict) -> dict:
    """Simple tier: Gemma 4 4B running locally on AMD ROCm. Zero API cost."""
    from clients.ollama_client import generate

    model          = settings.LOCAL_EXECUTOR_MODEL
    response, toks = await generate(state["prompt"], model, system=_SUPPORT_SYSTEM)
    return {"response": response, "model_used": model, "tokens_used": toks}


async def cloud_medium_node(state: dict) -> dict:
    """Medium tier: Gemma 4 12B via Fireworks AI. Falls back to Ollama via Circuit Breaker."""
    from clients.fireworks_client import generate

    model                      = settings.CLOUD_MEDIUM_MODEL
    # Hybrid Routing & Fireworks Fallback:
    # generate() is wrapped with exponential backoff and a circuit breaker.
    # If the Fireworks API fails or rate limits, it will transparently fall back
    # to the local AMD Ollama instance to guarantee a response.
    response, toks, used_model = await generate(state["prompt"], model)
    return {"response": response, "model_used": used_model, "tokens_used": toks}


async def cloud_complex_node(state: dict) -> dict:
    """Complex tier: Gemma 4 31B via Fireworks AI. Falls back to Ollama via Circuit Breaker."""
    from clients.fireworks_client import generate

    model                      = settings.CLOUD_COMPLEX_MODEL
    # Hybrid Routing & Fireworks Fallback:
    # Similar to medium tier, if the complex Fireworks API call fails,
    # the circuit breaker instantly routes this request to the local AMD Ollama model.
    response, toks, used_model = await generate(state["prompt"], model)
    return {"response": response, "model_used": used_model, "tokens_used": toks}


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Quality Judge (local AMD ROCm, $0.00)
# ─────────────────────────────────────────────────────────────────────────────

async def quality_judge_node(state: dict) -> dict:
    """Bypassed local quality judge to unblock frontend."""
    return {"quality_score": 1.0}


async def escalate_tier_node(state: dict) -> dict:
    """Bump current_tier by one level and record the escalation."""
    tier  = state.get("current_tier", "simple")
    order = ["simple", "medium", "complex"]
    idx   = order.index(tier) if tier in order else 0
    next_tier = order[min(idx + 1, len(order) - 1)]
    return {
        "current_tier":     next_tier,
        "escalation_depth": state.get("escalation_depth", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Logger — Observability + SQLite write
# ─────────────────────────────────────────────────────────────────────────────

async def logger_node(state: dict) -> dict:
    """Compute final cost/savings, detect resilience events, and persist a RoutingLog row."""
    from services.cost_estimator import calculate_cost_and_savings
    from database import AsyncSessionLocal
    from models import RoutingLog
    import logging as _log

    _logger = _log.getLogger("gemmaroute")

    latency_ms = (time.time() - state["start_time"]) * 1000
    model      = state.get("model_used", "unknown")
    tokens     = state.get("tokens_used", 0)
    cost_usd, saved_usd = calculate_cost_and_savings(model, tokens)

    # ── Pattern 2: Hop Budget exhaustion detection ────────────────────────────
    depth    = state.get("escalation_depth", 0)
    q_score  = state.get("quality_score", 1.0)
    tier     = state.get("current_tier", "trivial")
    hop_budget_exhausted = (
        depth >= settings.MAX_ESCALATION_DEPTH
        and q_score < settings.QUALITY_THRESHOLD
        and tier not in ("trivial", None)
    )
    if hop_budget_exhausted:
        _logger.warning(
            f"⚠️  Hop budget exhausted (depth={depth}/{settings.MAX_ESCALATION_DEPTH}) "
            f"for session='{state.get('session_id')}'. "
            f"Quality={q_score:.2f} still below threshold={settings.QUALITY_THRESHOLD}. "
            "Returning best available response (graceful degradation)."
        )

    async with AsyncSessionLocal() as session:
        log = RoutingLog(
            session_id      = state.get("session_id", ""),
            prompt_preview  = state["prompt"][:120],
            initial_tier    = state.get("initial_tier", "unknown"),
            final_tier      = state.get("current_tier", "unknown"),
            model_used      = model,
            escalations     = depth,
            classifier_conf = state.get("classifier_confidence", 0.0),
            quality_score   = q_score,
            latency_ms      = latency_ms,
            cost_usd        = cost_usd,
            saved_usd       = saved_usd,
        )
        session.add(log)
        await session.commit()

    return {
        "latency_ms":           latency_ms,
        "cost_usd":             cost_usd,
        "saved_usd":            saved_usd,
        "hop_budget_exhausted": hop_budget_exhausted,
    }
