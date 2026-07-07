"""
graph/edges.py — Conditional edge (router) functions for the LangGraph.

Each function inspects the current AgentState and returns a string key
that LangGraph uses to look up the next node in the path_map.

Resilience patterns:
  - Pattern 2 (Hop Budget): route_after_judge enforces a hard cap of
    MAX_ESCALATION_DEPTH hops before graceful degradation.
"""
from graph.state import AgentState
from config import settings

# ── Hop Budget constant (mirrors settings.MAX_ESCALATION_DEPTH) ──────────────
# Pattern 2: Hard cap on escalation hops. Even if the quality judge keeps
# scoring below threshold, the graph MUST terminate after this many tier upgrades
# to prevent unbounded loops. The best available response is returned as-is.
HOP_BUDGET = settings.MAX_ESCALATION_DEPTH   # default: 2


def route_after_filter(state: AgentState) -> str:
    """
    After the heuristic filter:
    - Trivial queries (hours, greetings) skip all LLM calls entirely.
    - Everything else goes to the Gemma classifier.
    """
    return "trivial_respond" if state.get("is_trivial") else "classifier"


def route_after_classifier(state: AgentState) -> str:
    """
    After the Gemma classifier assigns a tier, route to the matching executor.
    Returns: "simple" | "medium" | "complex"
    """
    return state.get("current_tier", "medium")


def route_after_judge(state: AgentState) -> str:
    """
    After the quality judge scores the response, decide: pass or escalate.

    Pattern 2 — Hop Budget:
      The graph may escalate at most HOP_BUDGET (2) times. After that, the
      best available response is returned regardless of quality score.
      This is a hard safety cap that prevents infinite escalation loops.

    Escalate conditions (ALL must be true):
      - quality_score is below the threshold
      - escalation_depth < HOP_BUDGET  (hop budget not yet exhausted)
      - current_tier is NOT "complex"  (can't escalate beyond complex)

    Otherwise → logger (graceful pass, even with low quality).
    """
    score = state.get("quality_score", 1.0)
    depth = state.get("escalation_depth", 0)
    tier  = state.get("current_tier", "complex")

    # Hard Hop Budget cap — any one of these terminates the escalation cycle
    hop_budget_exhausted = depth >= HOP_BUDGET
    at_ceiling           = tier == "complex"
    quality_acceptable   = score >= settings.QUALITY_THRESHOLD

    if quality_acceptable or hop_budget_exhausted or at_ceiling:
        return "logger"

    return "escalate"


def route_after_escalate(state: AgentState) -> str:
    """
    After bumping the tier, re-route to the appropriate executor.
    Returns the new current_tier value set by escalate_tier_node.
    """
    return state.get("current_tier", "complex")
