"""
graph/builder.py — Assembles and compiles the GemmaRoute LangGraph StateGraph.

The compiled `routing_graph` singleton is imported by route_endpoint.py
and called with `await routing_graph.ainvoke(initial_state)`.

Graph shape (see nodes.py for full docstring):
  START → filter → [trivial_respond → logger] | [classifier → executor → judge ⇄ escalate → logger] → END
"""
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from graph.nodes import (
    heuristic_filter_node,
    trivial_respond_node,
    gemma_classifier_node,
    local_executor_node,
    cloud_medium_node,
    cloud_complex_node,
    quality_judge_node,
    escalate_tier_node,
    logger_node,
)
from graph.edges import (
    route_after_filter,
    route_after_classifier,
    route_after_judge,
    route_after_escalate,
)


def build_routing_graph():
    builder = StateGraph(AgentState)

    # ── Register all nodes ───────────────────────────────────────────────────
    builder.add_node("filter",         heuristic_filter_node)
    builder.add_node("trivial_respond", trivial_respond_node)
    builder.add_node("classifier",     gemma_classifier_node)
    builder.add_node("simple",         local_executor_node)
    builder.add_node("medium",         cloud_medium_node)
    builder.add_node("complex",        cloud_complex_node)
    builder.add_node("judge",          quality_judge_node)
    builder.add_node("escalate",       escalate_tier_node)
    builder.add_node("logger",         logger_node)

    # ── Wire edges ───────────────────────────────────────────────────────────
    # Entry point
    builder.add_edge(START, "filter")

    # After filter: trivial shortcut OR full classify
    builder.add_conditional_edges(
        "filter",
        route_after_filter,
        {"trivial_respond": "trivial_respond", "classifier": "classifier"},
    )
    # Trivial path goes directly to logger (no LLM quality check needed)
    builder.add_edge("trivial_respond", "logger")

    # After classifier: route to matching executor tier
    builder.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {"simple": "simple", "medium": "medium", "complex": "complex"},
    )

    # All three executors feed into the quality judge
    builder.add_edge("simple",  "judge")
    builder.add_edge("medium",  "judge")
    builder.add_edge("complex", "judge")

    # After quality judge: pass or escalate (cycle-aware)
    builder.add_conditional_edges(
        "judge",
        route_after_judge,
        {"logger": "logger", "escalate": "escalate"},
    )

    # After escalation: re-execute at the new (higher) tier
    builder.add_conditional_edges(
        "escalate",
        route_after_escalate,
        {"simple": "simple", "medium": "medium", "complex": "complex"},
    )

    # Logger always terminates the graph
    builder.add_edge("logger", END)

    return builder.compile()


# ── Module-level singleton — compiled once at import time ────────────────────
routing_graph = build_routing_graph()
