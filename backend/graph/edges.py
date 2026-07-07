"""
graph/edges.py — Conditional edge (router) functions for the LangGraph.

Each function inspects the current AgentState and returns a string key
that LangGraph uses to look up the next node in the path_map.
"""
from graph.state import AgentState
from config import settings


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
    After the quality judge scores the response:
    - Pass (→ logger) if: score is good, max escalation depth reached, or
      already at the complex tier (can't go higher).
    - Escalate (→ escalate) otherwise.
    """
    score = state.get("quality_score", 1.0)
    depth = state.get("escalation_depth", 0)
    tier  = state.get("current_tier", "complex")

    should_pass = (
        score >= settings.QUALITY_THRESHOLD
        or depth >= settings.MAX_ESCALATION_DEPTH
        or tier == "complex"
    )
    return "logger" if should_pass else "escalate"


def route_after_escalate(state: AgentState) -> str:
    """
    After bumping the tier, re-route to the appropriate executor.
    Returns the new current_tier value set by escalate_tier_node.
    """
    return state.get("current_tier", "complex")
