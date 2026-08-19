"""FastAPI Application Entrypoint for The Secure LLM Gateway."""

import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse

from app.config import settings
from app.api.v1.routes import router as api_v1_router
from app.observability.metrics import metrics
from app.observability.logging import setup_structured_logging

logger = setup_structured_logging(debug=settings.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} on {settings.HOST}:{settings.PORT}...")
    logger.info(f"Semantic Caching enabled: {settings.ENABLE_SEMANTIC_CACHE} (Threshold: {settings.CACHE_SIMILARITY_THRESHOLD})")
    logger.info(f"Input Guardrails enabled: {settings.ENABLE_INJECTION_GUARDRAIL} (PII Scrubbing: {settings.ENABLE_PII_SCRUBBING})")
    logger.info(f"Resilient LLM Routing priority: {settings.PROVIDER_PRIORITY}")
    yield
    logger.info("Shutting down Secure LLM Gateway...")


app = FastAPI(
    title=settings.APP_NAME,
    description="A resilient, OpenAI-compatible proxy with semantic caching, input/output guardrails, fallback routing, and observability.",
    version="1.0.0",
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
        "semantic_cache": "active" if settings.ENABLE_SEMANTIC_CACHE else "disabled",
        "guardrails": "active" if settings.ENABLE_INJECTION_GUARDRAIL else "disabled",
    }


@app.get("/metrics")
async def prometheus_metrics():
    return PlainTextResponse(metrics.export_prometheus(), media_type="text/plain; version=0.0.4")


app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("app/static/index.html")
