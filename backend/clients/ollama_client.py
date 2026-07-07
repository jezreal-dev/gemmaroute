import json
import httpx
from config import settings

# ── Classifier system prompt ─────────────────────────────────────────────────
_CLASSIFY_SYSTEM = """You are a routing classifier for a customer support AI platform.

Analyse the customer message and classify it into exactly one tier:
- "simple"   → Basic FAQ, order tracking, payment info, simple yes/no, product info
- "medium"   → Account questions, return initiation, billing disputes, policy lookup
- "complex"  → Refund disputes, legal/compliance questions, multi-step escalations, SLA reviews

Respond ONLY with valid JSON — no markdown, no extra text, nothing else:
{"tier": "simple"|"medium"|"complex", "confidence": 0.0-1.0, "reasoning": "one concise sentence"}"""

# ── Judge prompt template ────────────────────────────────────────────────────
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


async def classify_prompt(prompt: str) -> dict:
    """
    Use Gemma 4 2B (LOCAL, AMD ROCm) to classify a prompt.
    Returns: {"tier": str, "confidence": float, "reasoning": str}
    Falls back to "medium" tier on any error to avoid dropped requests.
    """
    payload = {
        "model": settings.LOCAL_ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": _CLASSIFY_SYSTEM},
            {"role": "user", "content": f"Customer message: {prompt}"},
        ],
        "stream": False,
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_HOST}/api/chat", json=payload
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return json.loads(content)
    except (httpx.HTTPError, json.JSONDecodeError, KeyError):
        return {"tier": "medium", "confidence": 0.5, "reasoning": "fallback due to error"}


async def generate(prompt: str, model: str, system: str = "") -> tuple[str, int]:
    """
    Generate a response from a local Ollama model.
    Returns: (response_text, total_tokens_used)
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": model, "messages": messages, "stream": False}

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.OLLAMA_HOST}/api/chat", json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["message"]["content"]
        # Ollama returns prompt_eval_count (input) + eval_count (output)
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        return text, tokens


async def judge_quality(prompt: str, response: str) -> float:
    """
    Use Gemma 4 4B (LOCAL, AMD ROCm) as an LLM-as-judge.
    Returns a quality score float in [0.0, 1.0].
    Falls back to 0.5 on parse errors to allow the pipeline to continue.
    """
    payload = {
        "model": settings.LOCAL_JUDGE_MODEL,
        "messages": [
            {
                "role": "user",
                "content": _JUDGE_TEMPLATE.format(prompt=prompt, response=response),
            }
        ],
        "stream": False,
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_HOST}/api/chat", json=payload
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return float(json.loads(content).get("score", 0.5))
    except (httpx.HTTPError, json.JSONDecodeError, ValueError, KeyError):
        return 0.5


async def check_health() -> bool:
    """Ping Ollama to verify it is reachable. Used by GET /health."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.OLLAMA_HOST}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False
