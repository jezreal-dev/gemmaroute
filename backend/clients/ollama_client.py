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
# Classifier system prompt
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = """You are a routing classifier for a customer support AI platform.

Analyse the customer message and classify it into exactly one tier:
- "simple"   → Basic FAQ, order tracking, payment info, simple yes/no, product info
- "medium"   → Account questions, return initiation, billing disputes, policy lookup
- "complex"  → Refund disputes, legal/compliance questions, multi-step escalations, SLA reviews

Respond ONLY with valid JSON — no markdown, no extra text, nothing else:
{"tier": "simple"|"medium"|"complex", "confidence": 0.0-1.0, "reasoning": "one concise sentence"}"""


# ─────────────────────────────────────────────────────────────────────────────
# Quality judge prompt template
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_TEMPLATE = """You are a strict quality evaluator for customer support AI responses.

Customer Query:
{prompt}

AI Response:
{response}

Rate this response on four criteria (total 10 points):
1. Directly and completely answers the question (0–3 pts)
2. Is factually accurate and not misleading (0–3 pts)
3. Uses a professional, empathetic tone (0–2 pts)
4. Is concise without omitting key information (0–2 pts)

Respond ONLY with valid JSON — no markdown, no extra text:
{{"score": 0.0-1.0, "reason": "one sentence"}}

Where score = (your_total_points / 10). Be strict."""


# ─────────────────────────────────────────────────────────────────────────────
# Public API — all functions wrapped with exponential backoff
# ─────────────────────────────────────────────────────────────────────────────

@retry(**_ollama_retry)
async def classify_prompt(prompt: str) -> dict:
    """
    Use Gemma 4 2B (LOCAL, AMD ROCm) to classify a prompt into a routing tier.
    Returns: {"tier": str, "confidence": float, "reasoning": str}
    Falls back to "medium" tier on JSON parse errors.

    Retry: exponential backoff at 1s → 2s → 4s on transient HTTP errors.
    """
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
            return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Classifier returned invalid JSON — using medium fallback.")
        return {"tier": "medium", "confidence": 0.5, "reasoning": "JSON parse error fallback"}


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
        data   = resp.json()
        text   = data["message"]["content"]
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
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
