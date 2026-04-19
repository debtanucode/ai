from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Optional


class BaseCacheBackend(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        ...

    @abstractmethod
    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        ...


class MemoryCacheBackend(BaseCacheBackend):
    _store: dict[str, str] = {}

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        self._store[key] = value


class RedisCacheBackend(BaseCacheBackend):
    def __init__(self, redis_url: str) -> None:
        import redis.asyncio as aioredis
        self._client = aioredis.from_url(redis_url, decode_responses=True)

    async def get(self, key: str) -> Optional[str]:
        try:
            return await self._client.get(key)
        except Exception:
            return None

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        try:
            await self._client.setex(key, ttl, value)
        except Exception:
            pass


class CacheManager:
    def __init__(self, backend: BaseCacheBackend, context_ttl: int = 3600, response_ttl: int = 86400) -> None:
        self._backend = backend
        self._context_ttl = context_ttl
        self._response_ttl = response_ttl

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    async def get_context(self, knowledge_content: str) -> Optional[str]:
        key = "ctx:" + self._hash(knowledge_content)
        return await self._backend.get(key)

    async def set_context(self, knowledge_content: str, context: str, ttl: Optional[int] = None) -> None:
        key = "ctx:" + self._hash(knowledge_content)
        await self._backend.set(key, context, ttl or self._context_ttl)

    async def get_response(self, full_prompt: str) -> Optional[str]:
        key = "resp:" + self._hash(full_prompt)
        return await self._backend.get(key)

    async def set_response(self, full_prompt: str, response_json: str, ttl: Optional[int] = None) -> None:
        key = "resp:" + self._hash(full_prompt)
        await self._backend.set(key, response_json, ttl or self._response_ttl)
