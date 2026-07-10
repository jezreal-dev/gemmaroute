"""
tests/test_api.py — Integration tests for the FastAPI endpoints.

Tests use HTTPX AsyncClient with ASGI transport to call the app directly
(no live server needed). DB and LLM clients are mocked.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# All API calls require the X-API-Key header (except /health and /docs)
_HEADERS = {"X-API-Key": "gemmaroute-demo-2026"}


# ── Health endpoint (no auth required) ───────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_ok():
    with patch("routers.stats_endpoint.ollama_health", new=AsyncMock(return_value=True)):
        from httpx import AsyncClient, ASGITransport
        from main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["ollama"] == "reachable"


@pytest.mark.asyncio
async def test_health_ollama_unreachable():
    with patch("routers.stats_endpoint.ollama_health", new=AsyncMock(return_value=False)):
        from httpx import AsyncClient, ASGITransport
        from main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ollama"] == "unreachable"


# ── Auth middleware ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_route_rejects_missing_api_key():
    """Requests without X-API-Key must be rejected with 401."""
    from httpx import AsyncClient, ASGITransport
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/route", json={"prompt": "hello"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_route_rejects_wrong_api_key():
    from httpx import AsyncClient, ASGITransport
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/route", json={"prompt": "hello"}, headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_options_preflight_no_auth_required():
    """OPTIONS preflight must pass through auth middleware without 401.
    ASGI test transport may return 405 (method not allowed at route level)
    which is fine — the key is it must NOT return 401 (auth block)."""
    from httpx import AsyncClient, ASGITransport
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.options("/route", headers={"Origin": "https://gemmaroute.vercel.app"})
    # Must not be 401 (auth blocked) — 405 is acceptable (route-level, not auth)
    assert resp.status_code != 401, "OPTIONS should never return 401 (CORS preflight fix)"


# ── Trivial routing (no LLM call) ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_route_trivial_hours_query():
    """Business hours query must hit heuristic filter — zero LLM calls."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    with patch("database.AsyncSessionLocal", return_value=mock_session):
        from httpx import AsyncClient, ASGITransport
        from main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/route",
                json={"prompt": "What are your business hours?", "session_id": "pytest"},
                headers=_HEADERS,
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["routing"]["final_tier"] == "trivial"
    assert data["routing"]["estimated_cost_usd"] == 0.0
    assert data["routing"]["latency_ms"] >= 0
    assert "Monday" in data["response"]


@pytest.mark.asyncio
async def test_route_trivial_greeting():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    with patch("database.AsyncSessionLocal", return_value=mock_session):
        from httpx import AsyncClient, ASGITransport
        from main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/route",
                json={"prompt": "Hello!", "session_id": "pytest"},
                headers=_HEADERS,
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["routing"]["final_tier"] == "trivial"
    assert data["routing"]["model_used"] == "heuristic_filter"


# ── Route endpoint input validation ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_route_rejects_empty_prompt():
    from httpx import AsyncClient, ASGITransport
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/route", json={"prompt": ""}, headers=_HEADERS)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_route_rejects_missing_prompt():
    from httpx import AsyncClient, ASGITransport
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/route", json={}, headers=_HEADERS)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_route_rejects_prompt_too_long():
    from httpx import AsyncClient, ASGITransport
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/route", json={"prompt": "x" * 4097}, headers=_HEADERS)
    assert resp.status_code == 422


# ── Stats endpoint ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats_requires_auth():
    from httpx import AsyncClient, ASGITransport
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/stats")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_stats_returns_valid_schema():
    """Stats should return a valid dict even with an empty database."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.scalar = AsyncMock(return_value=0)

    mock_agg = MagicMock()
    mock_agg.one.return_value = MagicMock(
        total_cost=0, total_saved=0, avg_latency=0, avg_quality=0, avg_escalations=0
    )
    mock_dist = MagicMock()
    mock_dist.__iter__ = MagicMock(return_value=iter([]))
    mock_recent = MagicMock()
    mock_recent.scalars.return_value.all.return_value = []

    call_count = [0]
    async def mock_execute(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return mock_dist
        elif call_count[0] == 2:
            return mock_agg
        else:
            return mock_recent

    mock_session.execute = AsyncMock(side_effect=mock_execute)

    with patch("routers.stats_endpoint.AsyncSessionLocal", return_value=mock_session):
        from httpx import AsyncClient, ASGITransport
        from main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/stats", headers=_HEADERS)

    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "routing_distribution" in data
    assert "total_saved_vs_always_complex_usd" in data
    assert "recent_logs" in data
    assert isinstance(data["recent_logs"], list)
