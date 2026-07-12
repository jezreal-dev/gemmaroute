"""
main.py — FastAPI application factory for GemmaRoute.

Responsibilities:
  - Create the FastAPI app with metadata and lifespan
  - Register CORS middleware (allows dashboard to call the API)
  - Register request logging middleware
  - Mount both routers (route + stats)
  - Initialise the database on startup via the lifespan context
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import init_db
from routers.route_endpoint import router as route_router
from routers.stats_endpoint import router as stats_router

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gemmaroute")


# ── Lifespan: startup / shutdown hooks ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 GemmaRoute backend starting up...")
    await init_db()
    logger.info("✅ SQLite database initialised.")
    yield
    logger.info("🛑 GemmaRoute backend shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GemmaRoute",
    description=(
        "⚡ 3-Layer AMD-Native LLM Routing Engine.\n\n"
        "Routes customer support queries to the cheapest Gemma 4 model "
        "that meets the quality threshold — local AMD ROCm for free, "
        "Fireworks AI for heavier tasks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit dashboard and the Next.js frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gemmaroute.vercel.app",
        "https://gemmaroute-cyan.vercel.app",
        "http://localhost:3000",
        "http://localhost:8501",
    ],
    allow_origin_regex=r"https://gemmaroute.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ── Security middleware ───────────────────────────────────────────────────────
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Skip auth for CORS preflight requests — OPTIONS never carries API keys.
    # Without this, the browser's preflight gets a bare 401 before CORS headers
    # are attached, which the browser misreports as a CORS error.
    if request.method == "OPTIONS":
        response = await call_next(request)
        return response

    if request.url.path not in ["/docs", "/openapi.json", "/health", "/stats"]:
        api_key = request.headers.get("X-API-Key")
        if api_key != settings.API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    response = await call_next(request)
    return response


# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} [{ms:.1f}ms]"
    )
    return response


# ── Mount routers ─────────────────────────────────────────────────────────────
app.include_router(route_router)
app.include_router(stats_router)
