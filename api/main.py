"""
api/main.py
-----------
FastAPI Application Entrypoint for Fraud Risk Analytics & Detection System.
Implements lifespan artifact preloading, CORS security controls, request size limiting,
process timing middleware, and sanitized error handling.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.db import db_manager
from api.routes import router as api_router
from src.explainability.reason_codes import BusinessDecisionPolicy
from src.explainability.shap_explainer import FraudSHAPExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fraud_risk_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.
    Pre-loads Champion LightGBM booster, feature pipeline, and TreeSHAP explainer
    once at startup into app.state to ensure fast, low-latency inference.
    """
    logger.info("Initializing %s (v%s)...", settings.APP_NAME, settings.APP_VERSION)
    try:
        policy = BusinessDecisionPolicy(
            threshold_medium=settings.THRESHOLD_MEDIUM,
            threshold_high=settings.THRESHOLD_HIGH,
        )
        explainer = FraudSHAPExplainer(
            model_path=settings.MODEL_PATH,
            pipeline_path=settings.PIPELINE_PATH,
            policy=policy,
        )
        app.state.explainer = explainer
        logger.info("Champion LightGBM booster & TreeSHAP explainer loaded successfully.")
    except Exception as e:
        logger.error("Failed to pre-load model artifacts at startup: %s", str(e), exc_info=True)
        app.state.explainer = None

    yield

    logger.info("Shutting down %s...", settings.APP_NAME)
    if db_manager._pool:
        db_manager._pool.closeall()
        logger.info("Closed PostgreSQL connection pool.")


def create_app() -> FastAPI:
    """Factory creating configured FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Operational API for the Fraud Risk Analytics & Detection System. "
            "Provides simulated real-time scoring, TreeSHAP reason code extraction, "
            "held-out demo replay, and operational monitoring metrics."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Timing, Payload Size, and Rate Limiting Middleware
    _rate_limit_records: dict[str, list[float]] = {}

    @app.middleware("http")
    async def request_guard_and_timing_middleware(request: Request, call_next) -> Response:
        # Check payload size limit if Content-Length is provided
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.MAX_REQUEST_BODY_BYTES:
                    return JSONResponse(
                        status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                        content={
                            "detail": f"Request body exceeds maximum allowed size of {settings.MAX_REQUEST_BODY_BYTES} bytes."
                        },
                    )
            except ValueError:
                pass

        # Rate limiting on POST /predict (10 requests per minute per IP, bypass for test environment)
        client_ip = request.client.host if request.client else "unknown"
        is_test = settings.ENVIRONMENT == "test" or client_ip == "testclient"
        if request.method == "POST" and request.url.path == "/predict" and not is_test:
            now = time.time()
            timestamps = _rate_limit_records.get(client_ip, [])
            # Filter timestamps within the last 60 seconds
            timestamps = [t for t in timestamps if now - t < 60.0]
            if len(timestamps) >= 10:
                return JSONResponse(
                    status_code=getattr(status, "HTTP_429_TOO_MANY_REQUESTS", 429),
                    content={
                        "detail": "Rate limit exceeded. Please wait before scoring another transaction."
                    },
                )
            timestamps.append(now)
            _rate_limit_records[client_ip] = timestamps

        start_time = time.perf_counter()
        response: Response = await call_next(request)
        process_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        response.headers["X-Process-Time-Ms"] = str(process_time_ms)
        return response

    # 3. Global Exception Handler (Sanitizes internal server error output)
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        logger.error("Unhandled server exception on %s: %s", request.url.path, str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred while processing the request."},
        )

    # 4. Register API Routes
    app.include_router(api_router)

    # 5. Root Welcome Route
    @app.get("/", tags=["System"])
    async def root_info() -> dict[str, str]:
        return {
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "status": "online",
            "documentation": "/docs",
            "health_check": "/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
