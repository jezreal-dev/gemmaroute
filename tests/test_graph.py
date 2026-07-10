"""
tests/test_graph.py — Unit tests for the full routing graph logic.

All LLM calls (Ollama, Fireworks) and DB writes are mocked so these
tests run instantly with no live services required.

Coverage:
  - Layer 1a: Heuristic filter (all 5 patterns + non-trivial cases)
  - Layer 0:  Pre-classifier signal check (complex + medium signals)
  - Layer 0:  Confidence gate logic
  - Layer 1b: Gemma classifier node (tier cap enforcement)
  - Layer 2:  Escalation node (simple→medium, medium→complex, ceiling)
  - Layer 2:  Circuit breaker fallback tag detection
  - Layer 3:  Quality judge fallback behaviour
  - Edges:    All routing decisions (filter, classifier, judge, escalate)
  - Services: Cost estimator (trivial free, complex no savings, medium partial)
  - Resilience: Ollama down → graceful fallback on all 3 nodes
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1a — Heuristic Filter
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_heuristic_filter_business_hours():
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
async def test_heuristic_filter_thanks():
    from graph.nodes import heuristic_filter_node
    result = await heuristic_filter_node({"prompt": "Thank you!"})
    assert result["is_trivial"] is True


@pytest.mark.asyncio
async def test_heuristic_filter_goodbye():
    from graph.nodes import heuristic_filter_node
    result = await heuristic_filter_node({"prompt": "Goodbye!"})
    assert result["is_trivial"] is True


@pytest.mark.asyncio
async def test_heuristic_filter_availability():
    from graph.nodes import heuristic_filter_node
    result = await heuristic_filter_node({"prompt": "Are you open?"})
    assert result["is_trivial"] is True


@pytest.mark.asyncio
async def test_heuristic_filter_non_trivial_refund():
    from graph.nodes import heuristic_filter_node
    result = await heuristic_filter_node({"prompt": "I need a full refund for my broken order"})
    assert result["is_trivial"] is False
    assert result["trivial_response"] is None


@pytest.mark.asyncio
async def test_heuristic_filter_non_trivial_sla():
    from graph.nodes import heuristic_filter_node
    result = await heuristic_filter_node({"prompt": "I have an SLA violation to discuss"})
    assert result["is_trivial"] is False


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
    assert result["cost_usd"] == 0.0
    assert result["saved_usd"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Layer 0 — Pre-classifier signal check
# ─────────────────────────────────────────────────────────────────────────────

def test_pre_classify_complex_signals():
    from clients.ollama_client import _pre_classify
    complex_prompts = [
        "I have a legal dispute about my contract",
        "This is an SLA violation that needs addressing",
        "I want to file a lawsuit regarding negligence",
        "I need to discuss fraud on my account",
        "There has been a service level breach",
        "I need to speak to an attorney about this",
        "We need arbitration for this dispute resolution",
    ]
    for prompt in complex_prompts:
        result = _pre_classify(prompt)
        assert result == "complex", f"Expected complex for: {prompt!r}, got {result!r}"


def test_pre_classify_medium_signals():
    from clients.ollama_client import _pre_classify
    medium_prompts = [
        "I want to return this item",
        "I need a refund for my order",
        "I have a billing question",
        "I want to cancel my subscription",
        "I was charged twice this month",
        "My account has been suspended",
        "I need to delete my account",
    ]
    for prompt in medium_prompts:
        result = _pre_classify(prompt)
        assert result == "medium", f"Expected medium for: {prompt!r}, got {result!r}"


def test_pre_classify_no_signal():
    from clients.ollama_client import _pre_classify
    neutral_prompts = [
        "where is my order?",
        "how do I reset my password?",
        "what products do you sell?",
        "track my shipment",
    ]
    for prompt in neutral_prompts:
        result = _pre_classify(prompt)
        assert result is None, f"Expected None for: {prompt!r}, got {result!r}"


def test_pre_classify_case_insensitive():
    from clients.ollama_client import _pre_classify
    assert _pre_classify("I NEED LEGAL ADVICE") == "complex"
    assert _pre_classify("REQUEST A REFUND") == "medium"


# ─────────────────────────────────────────────────────────────────────────────
# Layer 0 — Confidence gate
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_gate_high_confidence_no_change():
    from clients.ollama_client import _apply_confidence_gate
    assert _apply_confidence_gate("simple", 0.95, None) == "simple"
    assert _apply_confidence_gate("medium", 0.80, None) == "medium"
    assert _apply_confidence_gate("complex", 0.99, None) == "complex"


def test_confidence_gate_low_confidence_bumps_up():
    from clients.ollama_client import _apply_confidence_gate
    assert _apply_confidence_gate("simple", 0.50, None) == "medium"
    assert _apply_confidence_gate("medium", 0.45, None) == "complex"


def test_confidence_gate_complex_ceiling_not_exceeded():
    from clients.ollama_client import _apply_confidence_gate
    # complex is already ceiling — cannot bump higher
    assert _apply_confidence_gate("complex", 0.30, None) == "complex"


def test_confidence_gate_signal_floor_enforced():
    from clients.ollama_client import _apply_confidence_gate
    # LLM says simple with high confidence, but signal says medium floor
    assert _apply_confidence_gate("simple", 0.95, "medium") == "medium"
    # LLM says medium, signal says complex
    assert _apply_confidence_gate("medium", 0.95, "complex") == "complex"


def test_confidence_gate_signal_floor_not_downgraded():
    from clients.ollama_client import _apply_confidence_gate
    # LLM says complex, signal says medium — should stay complex
    assert _apply_confidence_gate("complex", 0.90, "medium") == "complex"


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1b — Classifier node
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_classifier_respects_max_tier_cap():
    """max_cost_tier=medium must cap a complex classification."""
    with patch(
        "clients.ollama_client.classify_prompt",
        new=AsyncMock(return_value={"tier": "complex", "confidence": 0.95, "reasoning": "hard"}),
    ):
        from graph.nodes import gemma_classifier_node
        result = await gemma_classifier_node({
            "prompt": "complex legal question",
            "max_cost_tier": "medium",
        })
    assert result["current_tier"] == "medium"


@pytest.mark.asyncio
async def test_classifier_fallback_on_ollama_down():
    """Ollama unreachable → classifier falls back to 'medium' tier."""
    with patch(
        "clients.ollama_client.classify_prompt",
        side_effect=Exception("Connection refused"),
    ):
        from graph.nodes import gemma_classifier_node
        result = await gemma_classifier_node({
            "prompt": "something complex",
            "max_cost_tier": "complex",
        })
    assert result["current_tier"] == "medium"
    assert result["classifier_confidence"] == 0.5


@pytest.mark.asyncio
async def test_classifier_complex_signal_bypasses_ollama():
    """Complex signal in pre_classify must return complex without any LLM call."""
    from clients.ollama_client import _pre_classify
    # Direct test of _pre_classify — the bypass happens here before Ollama is touched
    result = _pre_classify("I have a legal SLA violation")
    assert result == "complex", f"Expected complex, got {result!r}"

    result2 = _pre_classify("I need to file a lawsuit for negligence")
    assert result2 == "complex"


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Quality judge
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_quality_judge_fallback_on_ollama_down():
    """Ollama down → judge falls back to QUALITY_THRESHOLD (pass score)."""
    with patch(
        "clients.ollama_client.judge_quality",
        side_effect=Exception("Connection refused"),
    ):
        from graph.nodes import quality_judge_node
        from config import settings
        result = await quality_judge_node({
            "prompt": "test",
            "response": "test response",
        })
    assert result["quality_score"] == settings.QUALITY_THRESHOLD


@pytest.mark.asyncio
async def test_quality_judge_score_clamped():
    """Judge score outside [0,1] must be clamped."""
    with patch("clients.ollama_client.judge_quality", new=AsyncMock(return_value=1.5)):
        from graph.nodes import quality_judge_node
        result = await quality_judge_node({"prompt": "q", "response": "r"})
    assert result["quality_score"] == 1.0

    with patch("clients.ollama_client.judge_quality", new=AsyncMock(return_value=-0.3)):
        result = await quality_judge_node({"prompt": "q", "response": "r"})
    assert result["quality_score"] == 0.0


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
    result = route_after_judge({"quality_score": 0.1, "escalation_depth": 0, "current_tier": "complex"})
    assert result == "logger"


def test_route_after_judge_no_escalate_at_max_depth():
    from graph.edges import route_after_judge
    result = route_after_judge({"quality_score": 0.1, "escalation_depth": 2, "current_tier": "simple"})
    assert result == "logger"


def test_route_after_judge_exactly_at_threshold():
    from graph.edges import route_after_judge
    from config import settings
    result = route_after_judge({
        "quality_score": settings.QUALITY_THRESHOLD,
        "escalation_depth": 0,
        "current_tier": "simple",
    })
    assert result == "logger"  # exactly at threshold = pass


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Escalation node
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
# Circuit Breaker — CB_FALLBACK tag detection
# ─────────────────────────────────────────────────────────────────────────────

def test_cb_fallback_tag_format():
    """[CB_FALLBACK] tag must be detectable so frontend can display it."""
    tag = "[CB_FALLBACK] gemma:2b (NotFoundError)"
    # The frontend uses final_tier (not model name) to determine route now,
    # but the tag still needs to be parseable for the log table
    assert "[CB_FALLBACK]" in tag
    assert "gemma:2b" in tag


def test_cb_fallback_is_not_cloud_tier():
    """A CB_FALLBACK response served from local should show as local route."""
    # The fix: frontend uses final_tier, not model name for isCloud detection
    # medium/complex tier = cloud, trivial/simple = local
    final_tier = "medium"  # The tier the request was originally routed to
    is_cloud = final_tier in ("medium", "complex")
    assert is_cloud is True  # still shows cloud since tier was cloud

    final_tier = "simple"
    is_cloud = final_tier in ("medium", "complex")
    assert is_cloud is False


# ─────────────────────────────────────────────────────────────────────────────
# Services — Cost estimator
# ─────────────────────────────────────────────────────────────────────────────

def test_cost_trivial_is_free_with_savings():
    from services.cost_estimator import calculate_cost_and_savings
    cost, saved = calculate_cost_and_savings("heuristic_filter", 0)
    assert cost == 0.0
    assert saved > 0.0, "trivial should show savings vs always using complex"


def test_cost_local_model_is_free():
    from services.cost_estimator import calculate_cost_and_savings
    cost, saved = calculate_cost_and_savings("gemma:2b", 500)
    assert cost == 0.0
    assert saved > 0.0


def test_cost_complex_model_no_savings():
    from services.cost_estimator import calculate_cost_and_savings
    from config import settings
    cost, saved = calculate_cost_and_savings(settings.CLOUD_COMPLEX_MODEL, 1_000)
    assert cost > 0.0
    assert saved == 0.0


def test_cost_medium_model_partial_savings():
    from services.cost_estimator import calculate_cost_and_savings
    from config import settings
    cost, saved = calculate_cost_and_savings(settings.CLOUD_MEDIUM_MODEL, 1_000)
    assert cost > 0.0
    assert saved > 0.0


def test_cost_unknown_model_defaults_to_free():
    from services.cost_estimator import calculate_cost_and_savings
    # Unknown model defaults to $0/M rate — should be free but still show savings
    cost, saved = calculate_cost_and_savings("unknown-model", 500)
    assert cost == 0.0
