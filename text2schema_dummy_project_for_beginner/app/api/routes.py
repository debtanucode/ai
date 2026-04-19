from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import (
    get_cache_manager,
    get_erd_generator,
    get_llm_router,
    get_retry_handler,
    get_schema_converter,
)
from app.models.schema import GenerateRequest, GenerateResponse, TargetDB

router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/dialects")
async def get_dialects() -> list[dict]:
    return [
        {"id": "postgresql", "label": "PostgreSQL", "output": "sql"},
        {"id": "mysql", "label": "MySQL", "output": "sql"},
        {"id": "mongodb", "label": "MongoDB", "output": "nosql"},
        {"id": "dynamodb", "label": "DynamoDB", "output": "nosql"},
    ]


@router.get("/providers")
async def get_providers(llm_router=Depends(get_llm_router)) -> list[dict]:
    return llm_router.get_available_providers()


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    retry_handler=Depends(get_retry_handler),
    converter=Depends(get_schema_converter),
    erd_gen=Depends(get_erd_generator),
    cache_manager=Depends(get_cache_manager),
) -> GenerateResponse:
    start = time.monotonic()

    # Cache check
    if request.use_cache:
        cache_key = f"{request.description}:{request.target_db}:{request.output_format}"
        cached_response = await cache_manager.get_response(cache_key)
        if cached_response:
            data = json.loads(cached_response)
            data["cached"] = True
            data["processing_time_ms"] = round((time.monotonic() - start) * 1000, 2)
            return GenerateResponse(**data)

    schema, quality, retry_count = await retry_handler.run(request)

    if schema is None:
        raise HTTPException(status_code=422, detail="Failed to generate valid schema after maximum retries")

    outputs = converter.convert(schema, request.output_format)
    erd_data = erd_gen.generate(schema)
    outputs["erd"] = json.dumps(erd_data)

    elapsed_ms = round((time.monotonic() - start) * 1000, 2)

    response = GenerateResponse(
        schema=schema,
        quality=quality,
        retry_count=retry_count,
        outputs=outputs,
        cached=False,
        processing_time_ms=elapsed_ms,
    )

    if request.use_cache:
        await cache_manager.set_response(
            f"{request.description}:{request.target_db}:{request.output_format}",
            response.model_dump_json(),
        )

    return response
