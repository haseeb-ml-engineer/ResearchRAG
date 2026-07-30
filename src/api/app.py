"""
FastAPI application entry point for ResearchRAG.

This module initializes the web application, configures middleware,
registers API routes, and establishes global exception handling. It
serves strictly as the transport layer, delegating all business
logic to the underlying pipelines and services via dependency injection.
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.logging_config import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage the application lifecycle (startup and shutdown).

    This context manager replaces the deprecated `@app.on_event("startup")`
    and `@app.on_event("shutdown")` decorators.

    Args:
        app: The FastAPI application instance.
    """
    logger.info("ResearchRAG API starting up. Initializing resources...")
    # Establish global resources (e.g., database connection pools) here.
    
    yield  # Application is running
    
    logger.info("ResearchRAG API shutting down. Cleaning up resources...")
    # Teardown and graceful disconnection of resources occurs here.


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application instance.

    Returns:
        A fully configured FastAPI application ready to be served by Uvicorn.
    """
    app = FastAPI(
        title="ResearchRAG API",
        description="Production-ready retrieval-augmented generation API.",
        version="1.0.0",
        lifespan=lifespan,
    )

    _configure_middleware(app)
    _configure_exception_handlers(app)
    _register_routers(app)

    return app


def _configure_middleware(app: FastAPI) -> None:
    """
    Attach middleware to the application.

    Args:
        app: The FastAPI application instance.
    """
    # 1. CORS Middleware
    # In a true production environment, allow_origins should be restricted
    # to the specific frontend domains configured in settings.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Request Logging and Processing Time Middleware
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
        """Log incoming requests, handle errors, and measure response times."""
        start_time = time.perf_counter()

        logger.info("Received request: %s %s", request.method, request.url.path)

        response: Response | None = None

        try:
            response = await call_next(request)
        except Exception as error:
            logger.error(
                "Unhandled exception during request processing: %s %s - %s",
                request.method,
                request.url.path,
                error,
            )
            raise  # Let the exception_handler catch it
        finally:
            process_time = time.perf_counter() - start_time
            
            status_code = 500
            if response is not None:
                response.headers["X-Process-Time"] = f"{process_time:.4f}"
                status_code = response.status_code

            logger.info(
                "Completed request: %s %s - Status: %d - Latency: %.4fs",
                request.method,
                request.url.path,
                status_code,
                process_time,
            )

        assert response is not None
        return response


def _configure_exception_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers to ensure consistent JSON responses.

    Args:
        app: The FastAPI application instance.
    """
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch all unhandled exceptions and return a standardized JSON format.
        This prevents raw stack traces from leaking to the client.
        """
        logger.error(
            "Global exception caught for %s %s: %s",
            request.method,
            request.url.path,
            str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred while processing the request.",
                "path": request.url.path,
            },
        )


def _register_routers(app: FastAPI) -> None:
    """
    Register all modular API routers with the main application.

    Args:
        app: The FastAPI application instance.
    """
    # Register the basic health check directly on the main app
    @app.get("/health", tags=["System"])
    async def health_check() -> dict:
        """
        Verify that the API is running and responsive.
        """
        return {
            "status": "healthy",
            "version": app.version,
            "timestamp": time.time(),
        }

    from src.api.routes import router as api_router
    app.include_router(api_router)
    
    logger.info("Routers configured successfully.")


# Instantiate the application for ASGI servers (e.g., Uvicorn)
app = create_app()
