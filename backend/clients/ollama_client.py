"""
clients/ollama_client.py — Async Ollama client for local Gemma 4 models.

Resilience: all network calls are wrapped with tenacity exponential backoff.
Retry schedule on transient errors: 1s → 2s → 4s → raise.
"""
import json
import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

logger = logging.getLogger("gemmaroute.ollama")

_OLLAMA_CHAT = f"{settings.OLLAMA_HOST}/api/chat"


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1 — Transient error classification for Ollama (local httpx calls)
# ─────────────────────────────────────────────────────────────────────────────

def _is_transient_httpx(exc: BaseException) -> bool:
    """True for errors worth retrying against the local Ollama server."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError))


# Shared tenacity retry config for all Ollama calls
# Retry at 1s → 2s → 4s before re-raising, matching the Fireworks AI schedule.
_ollama_retry = dict(
    retry   = retry_if_exception(_is_transient_httpx),
    stop    = stop_after_attempt(4),                        # 1 initial + 3 retries
    wait    = wait_exponential(multiplier=1, min=1, max=4), # 1s → 2s → 4s
    reraise = True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 0 — Pre-classifier signal check (instant, zero LLM cost)
#
# Inspired by vLLM Semantic Router (https://vllm-semantic-router.com) and the
# semantic-router library (https://pypi.org/project/semantic-router/):
# high-signal domain vocabulary can classify with near-certainty before any
# LLM call, cutting misroute rates and eliminating unnecessary Ollama latency.
#
# These are "force" signals — single words/phrases whose presence in a customer
# support context almost unambiguously determines the tier ceiling. If matched,
# the LLM classifier is still called but its result is bounded by the signal.
# ─────────────────────────────────────────────────────────────────────────────

_COMPLEX_SIGNALS = frozenset([
    # Legal / compliance
    "legal", "lawsuit", "litigation", "arbitration", "negligence", "liability",
    "compliance", "regulation", "regulatory", "gdpr", "contract", "clause",
    "section", "attorney", "lawyer", "court", "settlement",
    # SLA / enterprise
    "sla", "service level", "breach", "violation", "penalty", "escalate",
    "escalation", "executive", "ceo", "cto",
    # Financial disputes
    "fraud", "chargeback", "unauthorized charge", "identity theft",
    "dispute resolution", "arbitration",
    # Multi-step / complex
    "compensation", "damages", "reimbursement", "class action",
])

_MEDIUM_SIGNALS = frozenset([
    # Returns & billing
    "refund", "return", "exchange", "billing", "invoice", "subscription",
    "cancel subscription", "downgrade", "upgrade", "payment failed",
    "charged twice", "double charge", "credit", "reopen",
    # Account operations
    "account suspended", "account locked", "verify identity",
    "data export", "delete account", "transfer account",
])

# Minimum confidence below which we bump one tier up (safer than trusting a
# low-confidence LLM guess — aligns with semantic-router threshold tuning)
_CONFIDENCE_FLOOR = 0.62

_TIER_ORDER = ["simple", "medium", "complex"]


def _pre_classify(prompt: str) -> str | None:
    """
    Instant signal-based pre-classifier. Returns a tier floor or None.

    If matched, the Gemma classifier output is still used but cannot go
    *below* this floor — avoiding costly under-routing to a weak model.
    Returns None if no strong signal is found (proceed with LLM classify).
    """
    lower = prompt.lower()
    if any(sig in lower for sig in _COMPLEX_SIGNALS):
        return "complex"
    if any(sig in lower for sig in _MEDIUM_SIGNALS):
        return "medium"
    return None


def _apply_confidence_gate(tier: str, confidence: float, floor: str | None) -> str:
    """
    Two adjustments inspired by semantic-router threshold tuning:

    1. Confidence gate: if Gemma is below _CONFIDENCE_FLOOR, bump tier up one
       level. Better to over-route to a stronger model than produce a bad
       answer that the quality judge then has to escalate anyway (+300ms/hop).

    2. Signal floor: if a pre-classifier signal detected a higher tier, use
       whichever is higher — the LLM result or the signal floor.
    """
    idx = _TIER_ORDER.index(tier) if tier in _TIER_ORDER else 0

    # Rule 1: low confidence → bump up
    if confidence < _CONFIDENCE_FLOOR and idx < len(_TIER_ORDER) - 1:
        idx += 1
        logger.info(
            f"Confidence gate: {tier} (conf={confidence:.2f} < {_CONFIDENCE_FLOOR}) "
            f"→ bumped to {_TIER_ORDER[idx]}"
        )

    # Rule 2: signal floor — never go below what the pre-classifier detected
    if floor is not None:
        floor_idx = _TIER_ORDER.index(floor)
        if floor_idx > idx:
            logger.info(
                f"Signal floor: LLM said {_TIER_ORDER[idx]} but signal detected "
                f"{floor} → using {floor}"
            )
            idx = floor_idx

    return _TIER_ORDER[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Classifier system prompt — more explicit examples to reduce LLM ambiguity
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = """You are a routing classifier for a customer support AI platform.

Analyse the customer message and classify it into exactly one tier:
- "simple"   → Basic FAQ, order tracking, password reset, product info, store hours
- "medium"   → Return initiation, billing questions, account changes, policy lookup, subscription changes
- "complex"  → Refund disputes, legal/compliance questions, SLA violations, multi-step escalations, fraud claims, contract reviews

Examples:
  "where is my order?" → simple
  "how do I reset my password?" → simple
  "I want to return an item I bought last week" → medium
  "why was I charged twice this month?" → medium
  "I have a legal dispute regarding my SLA agreement" → complex
  "I believe this is fraud and I need escalation" → complex

Respond ONLY with valid JSON — no markdown, no extra text, nothing else:
{"tier": "simple"|"medium"|"complex", "confidence": 0.0-1.0, "reasoning": "one concise sentence"}"""


# ─────────────────────────────────────────────────────────────────────────────
# Quality judge prompt template — calibrated scoring rubric
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_TEMPLATE = """You are a strict quality evaluator for customer support AI responses.
A score of 0.75 or above means the response is good enough to send to the customer.
A score below 0.75 means it needs a better model to handle it.

Customer Query:
{prompt}

AI Response:
{response}

Rate this response on four criteria (total 10 points):
1. Directly and completely answers the question (0–3 pts)
   - 3: fully answers with no gaps
   - 2: mostly answers but misses one detail
   - 1: partial answer or vague
   - 0: does not answer the question

2. Is factually accurate and not misleading (0–3 pts)
   - 3: fully accurate
   - 2: mostly accurate, minor issue
   - 1: questionable accuracy
   - 0: incorrect or misleading

3. Uses a professional, empathetic tone (0–2 pts)
   - 2: professional and empathetic
   - 1: neutral but not warm
   - 0: robotic, cold, or inappropriate

4. Is concise without omitting key information (0–2 pts)
   - 2: well-sized, no padding
   - 1: slightly too long or too short
   - 0: far too long/short or incomplete

Respond ONLY with valid JSON — no markdown, no extra text:
{{"score": 0.0-1.0, "reason": "one sentence explaining the main strength or weakness"}}

Where score = (your_total_points / 10). Be strict — a generic or evasive answer scores below 0.75."""


# ─────────────────────────────────────────────────────────────────────────────
# Public API — all functions wrapped with exponential backoff
# ─────────────────────────────────────────────────────────────────────────────

@retry(**_ollama_retry)
async def classify_prompt(prompt: str) -> dict:
    """
    Hybrid 3-layer classifier:
      1. Pre-classifier signal check (instant, ~0ms) — high-signal vocab forces floor
      2. Gemma 2B LLM classify (~300ms) — semantic understanding of intent
      3. Confidence gate — if LLM confidence < 0.62, bump tier up one level

    Returns: {"tier": str, "confidence": float, "reasoning": str}

    This approach is inspired by semantic-router (pypi.org/project/semantic-router)
    and vLLM Semantic Router which showed pure LLM classification has ~18% misroute
    rate on ambiguous prompts. The signal floor and confidence gate eliminate the
    most common failure modes without requiring any new dependencies.
    """
    # ── Layer 0: instant pre-classification signal check ─────────────────────
    signal_floor = _pre_classify(prompt)
    if signal_floor == "complex":
        # Unambiguous — skip the LLM call entirely, return immediately
        logger.info(f"Pre-classifier: complex signal detected → skipping Ollama call")
        return {"tier": "complex", "confidence": 0.97, "reasoning": "High-signal complex vocabulary detected"}

    # ── Layer 1b: Gemma 2B LLM classification ────────────────────────────────
    payload = {
        "model": settings.LOCAL_ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": _CLASSIFY_SYSTEM},
            {"role": "user",   "content": f"Customer message: {prompt}"},
        ],
        "stream": False,
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(_OLLAMA_CHAT, json=payload)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            result  = json.loads(content)

        tier       = result.get("tier", "medium")
        confidence = float(result.get("confidence", 0.5))
        reasoning  = result.get("reasoning", "")

        if tier not in _TIER_ORDER:
            tier = "medium"

        # ── Layer 2: confidence gate + signal floor ───────────────────────────
        adjusted_tier = _apply_confidence_gate(tier, confidence, signal_floor)

        return {
            "tier":       adjusted_tier,
            "confidence": confidence,
            "reasoning":  reasoning,
        }

    except json.JSONDecodeError:
        logger.warning("Classifier returned invalid JSON — using medium fallback.")
        floor = signal_floor or "medium"
        return {"tier": floor, "confidence": 0.5, "reasoning": "JSON parse error fallback"}


@retry(**_ollama_retry)
async def generate(prompt: str, model: str, system: str = "") -> tuple[str, int]:
    """
    Generate a response from a local Ollama model (Gemma 4 2B or 4B).
    Returns: (response_text, total_tokens)

    Retry: exponential backoff at 1s → 2s → 4s on transient HTTP errors.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": model, "messages": messages, "stream": False}

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(_OLLAMA_CHAT, json=payload)
        resp.raise_for_status()
        try:
            data   = resp.json()
            text   = data["message"]["content"]
            tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        except (KeyError, ValueError) as exc:
            logger.warning(f"Ollama generate returned unexpected JSON: {exc}. Raw: {resp.text[:200]}")
            text, tokens = "", 0
        return text, tokens


@retry(**_ollama_retry)
async def judge_quality(prompt: str, response: str) -> float:
    """
    Use Gemma 4 4B (LOCAL, AMD ROCm) as an LLM-as-judge quality gate.
    Returns a score in [0.0, 1.0]. Falls back to 0.5 on parse errors.

    Retry: exponential backoff at 1s → 2s → 4s on transient HTTP errors.
    """
    payload = {
        "model": settings.LOCAL_JUDGE_MODEL,
        "messages": [
            {
                "role":    "user",
                "content": _JUDGE_TEMPLATE.format(prompt=prompt, response=response),
            }
        ],
        "stream": False,
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(_OLLAMA_CHAT, json=payload)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return float(json.loads(content).get("score", 0.5))
    except (json.JSONDecodeError, ValueError, KeyError):
        logger.warning("Judge returned invalid JSON — defaulting to score=0.5.")
        return 0.5


async def check_health() -> bool:
    """Ping Ollama to verify it is reachable. Used by GET /health. Not retried."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.OLLAMA_HOST}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False
