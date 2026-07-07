"""
tests/test_api.py — Integration tests for the FastAPI endpoints.

Tests use HTTPX AsyncClient with ASGI transport to call the app directly
(no live server needed). DB and LLM clients are mocked.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ── Health endpoint ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_ok():
    with patch("clients.ollama_client.check_health", new=AsyncMock(return_value=True)):
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
    with patch("clients.ollama_client.check_health", new=AsyncMock(return_value=False)):
        from httpx import AsyncClient, ASGITransport
        from main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ollama"] == "unreachable"


# ── Trivial routing (no LLM call) ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_route_trivial_hours_query():
    """Business hours query must hit heuristic filter — zero LLM calls."""
    # Mock DB write so we don't need a real SQLite file
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
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["routing"]["final_tier"] == "trivial"
    assert data["routing"]["estimated_cost_usd"] == 0.0
    assert data["routing"]["latency_ms"] >= 0
    assert "Monday" in data["response"]


# ── Route endpoint validation ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_route_rejects_empty_prompt():
    from httpx import AsyncClient, ASGITransport
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/route", json={"prompt": ""})
    assert resp.status_code == 422   # Pydantic validation error


@pytest.mark.asyncio
async def test_route_rejects_missing_prompt():
    from httpx import AsyncClient, ASGITransport
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/route", json={})
    assert resp.status_code == 422


# ── Stats endpoint (empty DB) ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats_returns_valid_schema():
    """Stats should return a valid dict even with an empty database."""
    from httpx import AsyncClient, ASGITransport
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "routing_distribution" in data
    assert "total_cost_usd" in data
    assert "recent_logs" in data
    assert isinstance(data["recent_logs"], list)
