"""
services/cost_estimator.py — Token cost math for routing observability.

Calculates the actual API cost for a request AND the hypothetical cost
of routing it to the most expensive (complex) model — giving us the
"saved_usd" metric shown on the dashboard.
"""
from config import settings, COST_PER_1M_TOKENS


# Average tokens for a typical customer support response (used for savings estimate
# when a local/trivial route handles a query that would have cost cloud tokens)
_AVG_CLOUD_TOKENS = 350


def calculate_cost_and_savings(model: str, tokens: int) -> tuple[float, float]:
    """
    Calculate actual cost and savings vs always using the complex model.

    Args:
        model:  The model ID that was actually used.
        tokens: Total tokens consumed (input + output).

    Returns:
        (actual_cost_usd, saved_vs_always_complex_usd)

    Both values are rounded to 8 decimal places for precision.

    NOTE: For local/trivial routes tokens may be 0. We still calculate savings
    using _AVG_CLOUD_TOKENS so the dashboard shows meaningful cost avoidance
    — the query WAS handled for free, so the complex-tier cost WAS avoided.
    """
    rate_actual  = COST_PER_1M_TOKENS.get(model, 0.0)
    actual_cost  = (tokens / 1_000_000) * rate_actual

    rate_complex      = COST_PER_1M_TOKENS.get(settings.CLOUD_COMPLEX_MODEL, 0.90)
    # Use actual tokens if available, otherwise use the average estimate
    comparison_tokens = tokens if tokens > 0 else _AVG_CLOUD_TOKENS
    hypothetical_cost = (comparison_tokens / 1_000_000) * rate_complex

    saved = max(0.0, hypothetical_cost - actual_cost)
    return round(actual_cost, 8), round(saved, 8)


def format_cost_usd(usd: float) -> str:
    """Human-readable cost string for display."""
    if usd == 0.0:
        return "$0.00 (free)"
    if usd < 0.001:
        return f"${usd * 1_000:.4f}m"   # millidollars
    return f"${usd:.6f}"
