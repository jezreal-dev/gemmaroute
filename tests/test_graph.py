"""
tests/test_graph.py — Unit tests for the routing graph logic.

All LLM calls (Ollama, Fireworks) and DB writes are mocked so these tests
run instantly with no live services required.
"""
import pytest
from unittest.mock import AsyncMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1a — Heuristic Filter
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_heuristic_filter_hours():
    from graph.nodes import heuristic_filter_node
    result = await heuristic_filter_node({"prompt": "What are your business hours?"})
    assert result["is_trivial"] is True
    assert "Monday" in result["trivial_response"]


@pytest.mark.asyncio
async def test_heuristic_filter_greeting():
    from graph.nodes import heuristic_filter_node
    result = await heuristic_filter_node({"prompt": "Hello!"})
    assert result["is_trivial"] is True


@pytest.mark.asyncio
async def test_heuristic_filter_non_trivial():
    from graph.nodes import heuristic_filter_node
    result = await heuristic_filter_node({"prompt": "I need a full refund for my broken order"})
    assert result["is_trivial"] is False
    assert result["trivial_response"] is None


@pytest.mark.asyncio
async def test_trivial_respond_node():
    from graph.nodes import trivial_respond_node
    result = await trivial_respond_node({
        "prompt":           "hi",
        "trivial_response": "Hello! How can I help?",
        "session_id":       "test",
    })
    assert result["response"] == "Hello! How can I help?"
    assert result["model_used"] == "heuristic_filter"
    assert result["quality_score"] == 1.0
    assert result["tokens_used"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Routing edges
# ─────────────────────────────────────────────────────────────────────────────

def test_route_after_filter_trivial():
    from graph.edges import route_after_filter
    assert route_after_filter({"is_trivial": True}) == "trivial_respond"


def test_route_after_filter_non_trivial():
    from graph.edges import route_after_filter
    assert route_after_filter({"is_trivial": False}) == "classifier"


def test_route_after_classifier_each_tier():
    from graph.edges import route_after_classifier
    for tier in ("simple", "medium", "complex"):
        assert route_after_classifier({"current_tier": tier}) == tier


def test_route_after_judge_passes_on_good_score():
    from graph.edges import route_after_judge
    result = route_after_judge({"quality_score": 0.9, "escalation_depth": 0, "current_tier": "simple"})
    assert result == "logger"


def test_route_after_judge_escalates_on_bad_score():
    from graph.edges import route_after_judge
    result = route_after_judge({"quality_score": 0.3, "escalation_depth": 0, "current_tier": "simple"})
    assert result == "escalate"


def test_route_after_judge_no_escalate_at_complex():
    from graph.edges import route_after_judge
    # Even with a bad score, complex is the ceiling — must log
    result = route_after_judge({"quality_score": 0.1, "escalation_depth": 0, "current_tier": "complex"})
    assert result == "logger"


def test_route_after_judge_no_escalate_at_max_depth():
    from graph.edges import route_after_judge
    # Max escalation depth reached — must log even with bad score
    result = route_after_judge({"quality_score": 0.1, "escalation_depth": 2, "current_tier": "simple"})
    assert result == "logger"


# ─────────────────────────────────────────────────────────────────────────────
# Escalation node
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_escalate_simple_to_medium():
    from graph.nodes import escalate_tier_node
    result = await escalate_tier_node({"current_tier": "simple", "escalation_depth": 0})
    assert result["current_tier"] == "medium"
    assert result["escalation_depth"] == 1


@pytest.mark.asyncio
async def test_escalate_medium_to_complex():
    from graph.nodes import escalate_tier_node
    result = await escalate_tier_node({"current_tier": "medium", "escalation_depth": 1})
    assert result["current_tier"] == "complex"
    assert result["escalation_depth"] == 2


@pytest.mark.asyncio
async def test_escalate_complex_stays_complex():
    from graph.nodes import escalate_tier_node
    result = await escalate_tier_node({"current_tier": "complex", "escalation_depth": 1})
    assert result["current_tier"] == "complex"


# ─────────────────────────────────────────────────────────────────────────────
# Classifier — tier cap enforcement
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_classifier_respects_max_tier_cap():
    """If classifier says 'complex' but max_cost_tier is 'medium', cap to medium."""
    with patch(
        "clients.ollama_client.classify_prompt",
        new=AsyncMock(return_value={"tier": "complex", "confidence": 0.95, "reasoning": "hard question"}),
    ):
        from graph.nodes import gemma_classifier_node
        result = await gemma_classifier_node({
            "prompt": "difficult question",
            "max_cost_tier": "medium",
        })
    assert result["current_tier"] == "medium"
    assert result["classifier_confidence"] == 0.95


# ─────────────────────────────────────────────────────────────────────────────
# Cost estimator
# ─────────────────────────────────────────────────────────────────────────────

def test_cost_local_model_is_free():
    from services.cost_estimator import calculate_cost_and_savings
    cost, saved = calculate_cost_and_savings("gemma4:4b", 500)
    assert cost == 0.0
    assert saved > 0.0   # saved something vs complex


def test_cost_complex_model_zero_savings():
    from services.cost_estimator import calculate_cost_and_savings
    from config import settings
    cost, saved = calculate_cost_and_savings(settings.CLOUD_COMPLEX_MODEL, 1_000)
    assert cost > 0.0
    assert saved == 0.0   # no savings when using the most expensive model


def test_cost_medium_model_partial_savings():
    from services.cost_estimator import calculate_cost_and_savings
    from config import settings
    cost, saved = calculate_cost_and_savings(settings.CLOUD_MEDIUM_MODEL, 1_000)
    assert cost > 0.0
    assert saved > 0.0   # medium is cheaper than complex
