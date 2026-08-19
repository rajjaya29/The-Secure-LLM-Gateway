"""FastAPI Application Entrypoint for The Secure LLM Gateway."""

import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse

from app.config import settings
from app.api.v1.routes import router as api_v1_router
from app.observability.metrics import metrics
from app.observability.logging import setup_structured_logging
from app.observability.database import sqlite_logger
from app.router.llm_router import LLMRouter

logger = setup_structured_logging(debug=settings.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} on {settings.HOST}:{settings.PORT}...")
    logger.info(f"ChromaDB Semantic Cache enabled: {settings.ENABLE_SEMANTIC_CACHE} (Threshold: {settings.CACHE_SIMILARITY_THRESHOLD})")
    logger.info(f"API-Key Authentication: {settings.REQUIRE_API_KEY} (Sliding-Window: {settings.RATE_LIMIT_MAX_REQUESTS} req / {settings.RATE_LIMIT_WINDOW_SECONDS}s)")
    logger.info(f"SQLite Structured Logging: {settings.SQLITE_DB_PATH}")
    yield
    logger.info("Shutting down Secure LLM Gateway...")


app = FastAPI(
    title=settings.APP_NAME,
    description="An authenticated, rate-limited LLM reverse-proxy in FastAPI with ChromaDB semantic caching, sliding-window rate limiting, and SQLite analytics.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    t_start = time.perf_counter()
    metrics.p_active_requests.inc()

    try:
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Gateway-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response
    finally:
        metrics.p_active_requests.dec()


app.include_router(api_v1_router, prefix="/v1")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "gateway": settings.APP_NAME,
        "semantic_cache": "active (ChromaDB + all-MiniLM-L6-v2)",
        "guardrails": "active",
        "auth": "X-API-Key required" if settings.REQUIRE_API_KEY else "disabled",
    }


@app.get("/stats")
async def root_stats():
    """Live analytics metrics endpoint tracking cache-hit ratios, latency distributions, and per-key usage."""
    sqlite_stats = await sqlite_logger.get_analytics()
    return sqlite_stats


@app.get("/metrics")
async def prometheus_metrics():
    return PlainTextResponse(metrics.export_prometheus(), media_type="text/plain; version=0.0.4")


app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")
