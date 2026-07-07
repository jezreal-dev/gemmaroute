"""
routers/stats_endpoint.py — GET /stats and GET /health

Provides aggregate routing statistics for the Streamlit dashboard
and a health check endpoint for Docker compose health checks.
"""
from typing import Any

from fastapi import APIRouter
from sqlalchemy import select, func

from database import AsyncSessionLocal
from models import RoutingLog
from clients.ollama_client import check_health as ollama_health

router = APIRouter(tags=["observability"])


@router.get("/health", summary="Health check — returns status of all services")
async def health() -> dict[str, str]:
    ollama_ok = await ollama_health()
    return {
        "status": "ok",
        "ollama": "reachable" if ollama_ok else "unreachable",
        "db":     "ok",
    }


@router.get("/stats", summary="Aggregate routing statistics for the dashboard")
async def get_stats() -> dict[str, Any]:
    async with AsyncSessionLocal() as session:

        # ── Total request count ───────────────────────────────────────────────
        total = await session.scalar(
            select(func.count()).select_from(RoutingLog)
        ) or 0

        # ── Routing distribution (final_tier → count) ─────────────────────────
        dist_rows = await session.execute(
            select(RoutingLog.final_tier, func.count().label("cnt"))
            .group_by(RoutingLog.final_tier)
        )
        distribution = {row.final_tier: row.cnt for row in dist_rows}

        # ── Aggregate metrics ─────────────────────────────────────────────────
        agg = await session.execute(
            select(
                func.sum(RoutingLog.cost_usd).label("total_cost"),
                func.sum(RoutingLog.saved_usd).label("total_saved"),
                func.avg(RoutingLog.latency_ms).label("avg_latency"),
                func.avg(RoutingLog.quality_score).label("avg_quality"),
                func.avg(RoutingLog.escalations).label("avg_escalations"),
            )
        )
        row = agg.one()

        # ── 20 most recent logs ───────────────────────────────────────────────
        recent_q = await session.execute(
            select(RoutingLog).order_by(RoutingLog.created_at.desc()).limit(20)
        )
        recent = recent_q.scalars().all()

        recent_list = [
            {
                "id":             r.id,
                "session_id":     r.session_id,
                "prompt_preview": r.prompt_preview,
                "initial_tier":   r.initial_tier,
                "final_tier":     r.final_tier,
                "model_used":     r.model_used,
                "escalations":    r.escalations,
                "quality_score":  round(r.quality_score, 3),
                "latency_ms":     round(r.latency_ms, 1),
                "cost_usd":       round(r.cost_usd, 8),
                "saved_usd":      round(r.saved_usd, 8),
                "created_at":     r.created_at.isoformat() if r.created_at else None,
            }
            for r in recent
        ]

    return {
        "total_requests":                      int(total),
        "routing_distribution":                distribution,
        "total_cost_usd":                      round(float(row.total_cost or 0), 6),
        "total_saved_vs_always_complex_usd":   round(float(row.total_saved or 0), 6),
        "avg_latency_ms":                      round(float(row.avg_latency or 0), 1),
        "avg_quality_score":                   round(float(row.avg_quality or 0), 3),
        "escalation_rate":                     round(float(row.avg_escalations or 0), 3),
        "recent_logs":                         recent_list,
    }
