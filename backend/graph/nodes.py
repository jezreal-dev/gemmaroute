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

# ── Layer 0 — Pre-Classifier Signal Keywords ─────────────────────────────────
# High-signal vocabulary that bypasses the Gemma classifier entirely.
# Detected in ~0ms — sets a minimum tier floor before any LLM runs.
_COMPLEX_SIGNALS  = {"legal", "sla", "violation", "contract", "compliance", "lawsuit", "fraud", "escalate"}
_MEDIUM_SIGNALS   = {"return", "refund", "billing", "charge", "invoice", "dispute", "cancel", "chargeback"}


async def gemma_classifier_node(state: dict) -> dict:
    """
    Layer 1b: Use local Gemma 2B via Ollama to classify the prompt into a tier.
    Falls back to 'medium' tier if Ollama is unreachable (e.g. no local GPU).
    Zero API cost — runs entirely on local AMD ROCm hardware.

    Layer 0 pre-check: high-signal keywords are detected instantly (~0ms) and
    bypass the LLM classifier entirely, setting a minimum tier floor.
    """
    import logging as _log
    _logger = _log.getLogger("gemmaroute")

    # ── Layer 0: Pre-Classifier Signal Check (~0ms, $0.00) ───────────────────
    prompt_lower = state["prompt"].lower()
    prompt_words = set(re.findall(r"\b\w+\b", prompt_lower))

    complex_hits = prompt_words & _COMPLEX_SIGNALS
    medium_hits  = prompt_words & _MEDIUM_SIGNALS

    if complex_hits:
        _logger.info(
            f"⚡ Signal bypass: complex keywords detected {complex_hits} — "
            "skipping classifier, routing directly to complex tier."
        )
        return {
            "initial_tier":          "complex",
            "current_tier":          "complex",
            "classifier_confidence": 0.97,
        }

    if medium_hits:
        _logger.info(
            f"⚡ Signal bypass: medium keywords detected {medium_hits} — "
            "skipping classifier, routing directly to medium tier."
        )
        return {
            "initial_tier":          "medium",
            "current_tier":          "medium",
            "classifier_confidence": 0.90,
        }

    # ── Layer 1b: Gemma 2B Semantic Classifier ───────────────────────────────
    try:
        from clients.ollama_client import classify_prompt
        result = await classify_prompt(state["prompt"])
        tier       = result.get("tier", "medium")
        confidence = float(result.get("confidence", 0.5))
        # Clamp tier to valid values
        if tier not in ("simple", "medium", "complex"):
            tier = "medium"
        # ── Enforce max_cost_tier cap ─────────────────────────────────────────
        # If the caller set a cost ceiling (e.g. max_cost_tier="medium"),
        # never route above it regardless of classifier output.
        max_tier   = state.get("max_cost_tier", "complex")
        tier_order = ["simple", "medium", "complex"]
        if max_tier in tier_order and tier_order.index(tier) > tier_order.index(max_tier):
            _logger.info(f"Capping tier {tier} → {max_tier} (max_cost_tier enforced)")
            tier = max_tier
        _logger.info(f"Classifier → tier={tier} confidence={confidence:.2f}")
    except Exception as exc:
        # Ollama unreachable or timed out — safe fallback to medium tier
        _logger.warning(
            f"⚠️  Classifier failed ({type(exc).__name__}): {exc}. "
            "Falling back to 'medium' tier."
        )
        tier, confidence = "medium", 0.5

    return {
        "initial_tier":          tier,
        "current_tier":          tier,
        "classifier_confidence": confidence,
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
    """
    Layer 3: Use local Gemma 2B via Ollama as an LLM-as-judge quality gate.
    Scores the response 0.0–1.0. Falls back to 0.75 (pass threshold) if Ollama
    is unreachable — ensures the pipeline keeps flowing without a local GPU.
    Zero API cost — runs entirely on local AMD ROCm hardware.
    """
    import logging as _log
    _logger = _log.getLogger("gemmaroute")

    try:
        from clients.ollama_client import judge_quality
        score = await judge_quality(
            prompt   = state["prompt"],
            response = state.get("response", ""),
        )
        score = max(0.0, min(1.0, float(score)))   # clamp to [0, 1]
        _logger.info(f"Quality judge → score={score:.2f} (threshold={settings.QUALITY_THRESHOLD})")
    except Exception as exc:
        # Ollama unreachable — default to passing score so pipeline doesn't stall
        _logger.warning(
            f"⚠️  Quality judge failed ({type(exc).__name__}): {exc}. "
            f"Defaulting to score={settings.QUALITY_THRESHOLD} (pass)."
        )
        score = settings.QUALITY_THRESHOLD

    return {"quality_score": score}


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
