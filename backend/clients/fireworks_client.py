"""
clients/fireworks_client.py — Async Fireworks AI client.

Uses the OpenAI Python SDK pointed at Fireworks AI's inference endpoint.
Fireworks AI is fully OpenAI-API-compatible, so this is a drop-in client
with only base_url and api_key customized.
"""
from openai import AsyncOpenAI
from config import settings

# Module-level singleton — created on first call, reused for all subsequent calls
_client: AsyncOpenAI | None = None

_DEFAULT_SYSTEM = (
    "You are a professional, empathetic customer support agent. "
    "Provide accurate, thorough, and actionable responses. "
    "Acknowledge when you lack specific company information and offer to escalate."
)


def _get_client() -> AsyncOpenAI:
    """Return the module-level Fireworks AI client, creating it if needed."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=settings.FIREWORKS_BASE_URL,
            api_key=settings.FIREWORKS_API_KEY,
        )
    return _client


async def generate(
    prompt: str,
    model: str,
    system: str = _DEFAULT_SYSTEM,
) -> tuple[str, int]:
    """
    Generate a response from Fireworks AI (Gemma 4 12B or 31B).
    Returns: (response_text, total_tokens_used)
    """
    client = _get_client()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    text   = resp.choices[0].message.content or ""
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text, tokens
