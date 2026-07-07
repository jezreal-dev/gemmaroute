"""
database.py — Async SQLAlchemy engine, session factory, and DB initialiser.

Uses aiosqlite as the SQLite driver so all DB operations are non-blocking
inside FastAPI's async event loop.
"""
import os
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase
from config import settings

# Ensure the data directory exists for local development
os.makedirs("data", exist_ok=True)

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Shared declarative base — all ORM models inherit from this."""
    pass


async def get_db():
    """FastAPI dependency injection — yields a managed async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """
    Create all tables on application startup.
    Called inside the FastAPI lifespan context manager.
    """
    async with engine.begin() as conn:
        from models import RoutingLog  # noqa: F401 — registers model with Base.metadata
        await conn.run_sync(Base.metadata.create_all)
