"""Health check endpoint."""
from __future__ import annotations
import httpx
from fastapi import APIRouter
from ...config import settings

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health_check() -> dict:
    ollama_ok = False
    ollama_detail = "unavailable"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.ollama_url}/api/tags")
            ollama_ok = resp.status_code == 200
            ollama_detail = "ok" if ollama_ok else f"status {resp.status_code}"
    except Exception as exc:
        ollama_detail = str(exc)

    return {
        "status": "ok",
        "ollama": {"available": ollama_ok, "detail": ollama_detail, "url": settings.ollama_url},
        "model": settings.ollama_model,
    }
