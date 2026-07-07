"""
clients/fireworks_client.py — Async Fireworks AI client.

Resilience patterns implemented here:
  1. Exponential Backoff (tenacity): retries transient errors at 1s → 2s → 4s
     before declaring a request failed.
  2. Circuit Breaker: trips OPEN after 3 consecutive failures; stays open for 60s,
     then auto-resets to CLOSED. Prevents thundering-herd against a down API.
  3. Ollama Fallback: while the circuit is OPEN, all traffic is transparently
     re-routed to the local AMD gemma4:4b model. The model_used field in the
     routing log is tagged "[CB_FALLBACK]" so the dashboard shows it clearly.
"""
import asyncio
import logging
import time

import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

logger = logging.getLogger("gemmaroute.circuit_breaker")


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 2 — Circuit Breaker
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Asyncio-safe 2-state Circuit Breaker: CLOSED (normal) → OPEN (fallback).

    Transitions:
      CLOSED → OPEN : after `failure_threshold` consecutive errors
      OPEN → CLOSED : automatically after `reset_timeout` seconds (time-based)

    Thread-safety: uses asyncio.Lock so it's safe in FastAPI's event loop.
    """

    CLOSED = "CLOSED"
    OPEN   = "OPEN"

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout     = reset_timeout
        self._state            = self.CLOSED
        self._failure_count    = 0
        self._tripped_at: float | None = None
        self._lock             = asyncio.Lock()

    @property
    def state(self) -> str:
        """Current state — auto-recovers OPEN → CLOSED after timeout expires."""
        if self._state == self.OPEN:
            elapsed = time.time() - (self._tripped_at or time.time())
            if elapsed >= self.reset_timeout:
                self._state         = self.CLOSED
                self._failure_count = 0
                self._tripped_at    = None
                logger.info(
                    "🟢 Circuit CLOSED — Fireworks AI timeout elapsed. "
                    "Resuming normal cloud routing."
                )
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == self.OPEN

    @property
    def seconds_until_reset(self) -> float:
        if not self.is_open or self._tripped_at is None:
            return 0.0
        return max(0.0, self.reset_timeout - (time.time() - self._tripped_at))

    async def record_failure(self) -> bool:
        """
        Record one consecutive failure.
        Returns True if this failure was the one that tripped the circuit open.
        """
        async with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold and self._state == self.CLOSED:
                self._state      = self.OPEN
                self._tripped_at = time.time()
                logger.warning(
                    f"🔴 CIRCUIT BREAKER TRIPPED — {self._failure_count} consecutive "
                    f"Fireworks AI failures. Falling back to local Ollama for "
                    f"{self.reset_timeout:.0f}s. "
                    f"Dashboard will show [CB_FALLBACK] tags in the routing log."
                )
                return True
            return False

    async def record_success(self) -> None:
        async with self._lock:
            if self._failure_count > 0:
                logger.debug("✅ Fireworks AI call succeeded — resetting failure counter.")
            self._failure_count = 0
            self._state         = self.CLOSED
            self._tripped_at    = None

    def status_dict(self) -> dict:
        """Serializable status for the /health endpoint."""
        return {
            "state":             self.state,
            "failure_count":     self._failure_count,
            "failure_threshold": self.failure_threshold,
            "seconds_until_reset": round(self.seconds_until_reset, 1),
        }


# Module-level singletons
circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=60.0)
_client: AsyncOpenAI | None = None


def get_circuit_state() -> dict:
    """Public accessor used by GET /health to expose CB state to the dashboard."""
    return circuit_breaker.status_dict()


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=settings.FIREWORKS_BASE_URL,
            api_key=settings.FIREWORKS_API_KEY,
        )
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1 — Transient error classification (what tenacity should retry)
# ─────────────────────────────────────────────────────────────────────────────

def _is_transient_fireworks(exc: BaseException) -> bool:
    """
    True for errors worth retrying (rate limits, server errors, network blips).
    False for auth/validation errors — don't waste retries on those.
    """
    if isinstance(exc, openai.RateLimitError):          # HTTP 429
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500                   # HTTP 5xx
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True                                     # Network-level failures
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1 — Inner API call with Tenacity Exponential Backoff
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    retry   = retry_if_exception(_is_transient_fireworks),
    stop    = stop_after_attempt(4),                        # 1 initial + 3 retries
    wait    = wait_exponential(multiplier=1, min=1, max=4), # 1s → 2s → 4s
    reraise = True,                                         # raise after all attempts
)
async def _api_call_with_backoff(
    model: str,
    messages: list,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    """
    Wrapped Fireworks API call.
    Tenacity retries this on transient errors with exponential backoff:
      Attempt 1 (immediate) → wait 1s → Attempt 2 → wait 2s →
      Attempt 3 → wait 4s → Attempt 4 → raise → Circuit Breaker records failure.
    """
    client = _get_client()
    resp = await client.chat.completions.create(
        model       = model,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
    )
    text   = resp.choices[0].message.content or ""
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text, tokens


# ─────────────────────────────────────────────────────────────────────────────
# Public generate() — Circuit Breaker gate + Backoff wrapper + Ollama Fallback
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_SYSTEM = (
    "You are a professional, empathetic customer support agent. "
    "Provide accurate, thorough, and actionable responses. "
    "Acknowledge when you lack specific company information and offer to escalate."
)


async def generate(
    prompt: str,
    model: str,
    system: str = _DEFAULT_SYSTEM,
) -> tuple[str, int, str]:
    """
    Generate a response via Fireworks AI.

    Resilience flow:
      1. If circuit is OPEN → immediately fall back to Ollama (no wasted time)
      2. If circuit is CLOSED → attempt API call with exponential backoff (1s/2s/4s)
      3. If all retries fail   → record failure in circuit breaker, fall back to Ollama

    Returns: (response_text, total_tokens, actual_model_used)
      - actual_model_used is the original model ID on success
      - actual_model_used is "[CB_FALLBACK] gemma4:4b (reason: ...)" on fallback
        so it's immediately visible in the dashboard routing log.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]

    # ── Gate: skip API call entirely if circuit is already open ──────────────
    if circuit_breaker.is_open:
        ttl = circuit_breaker.seconds_until_reset
        logger.warning(
            f"🔴 Circuit OPEN — skipping Fireworks API call. "
            f"Routing to local Ollama fallback (resets in {ttl:.0f}s)."
        )
        return await _ollama_fallback(prompt, reason="circuit_open")

    # ── Attempt Fireworks call with tenacity backoff ──────────────────────────
    try:
        text, tokens = await _api_call_with_backoff(model, messages, 1024, 0.3)
        await circuit_breaker.record_success()
        return text, tokens, model

    except Exception as exc:
        tripped = await circuit_breaker.record_failure()
        count   = circuit_breaker._failure_count

        if tripped:
            logger.error(
                f"🔴 Circuit BREAKER TRIPPED by {type(exc).__name__} "
                f"(after {count} failures). "
                "All Fireworks traffic → Ollama for next 60s. "
                "Look for [CB_FALLBACK] tags in the routing log."
            )
        else:
            logger.warning(
                f"⚠️  Fireworks failure {count}/{circuit_breaker.failure_threshold}: "
                f"{type(exc).__name__}. Falling back to Ollama for this request."
            )

        return await _ollama_fallback(prompt, reason=type(exc).__name__)


async def _ollama_fallback(prompt: str, reason: str) -> tuple[str, int, str]:
    """
    Route to local Ollama when Fireworks AI is unavailable.
    The tagged model name surfaces as a visible alert in the dashboard routing log.
    """
    from clients.ollama_client import generate as ollama_generate

    fallback_model = settings.LOCAL_EXECUTOR_MODEL
    fallback_system = (
        "You are a professional customer support agent. "
        "Note: operating in resilience fallback mode — provide your best response."
    )
    try:
        text, tokens = await ollama_generate(prompt, fallback_model, system=fallback_system)
    except Exception as e:
        logger.error(f"Ollama fallback also failed: {e}")
        text   = (
            "I'm experiencing technical difficulties right now. "
            "Please try again shortly or contact our support team directly."
        )
        tokens = 0

    tagged_model = f"[CB_FALLBACK] {fallback_model} ({reason})"
    return text, tokens, tagged_model
