"""FastAPI application factory with lifespan management."""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..db.migrations import create_tables
from .routes import evaluate, results, health, websocket


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    await create_tables()
    try:
        import nltk
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
    except Exception:
        pass
    yield
    # Shutdown (nothing to clean up for SQLite/in-memory cache)


def create_app() -> FastAPI:
    app = FastAPI(
        title="SchemaEval",
        description="JSON Output Quality Evaluation Framework",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(evaluate.router)
    app.include_router(results.router)
    app.include_router(websocket.router)

    return app


app = create_app()
