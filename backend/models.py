"""
models.py — SQLAlchemy ORM model for routing decision logs.

Every request that passes through the GemmaRoute engine produces one
RoutingLog row, capturing the full lifecycle: tier classification, model
selection, escalation depth, quality score, latency, and cost metrics.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from database import Base


class RoutingLog(Base):
    __tablename__ = "routing_logs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    session_id     = Column(String(64),  nullable=True, index=True)
    prompt_preview = Column(Text)               # First 120 chars of prompt
    initial_tier   = Column(String(16))         # trivial | simple | medium | complex
    final_tier     = Column(String(16))         # Tier that produced the accepted response
    model_used     = Column(String(128))        # Exact model identifier
    escalations    = Column(Integer, default=0) # Number of tier upgrades triggered
    classifier_conf = Column(Float, default=0.0) # Gemma classifier confidence [0,1]
    quality_score  = Column(Float, default=0.0)  # LLM-as-judge score [0,1]
    latency_ms     = Column(Float)              # End-to-end request latency
    cost_usd       = Column(Float, default=0.0) # Actual API cost in USD
    saved_usd      = Column(Float, default=0.0) # Cost saved vs always routing to complex
    created_at     = Column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
