from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

from app.models.schema import GenerateRequest, QualityScore, SchemaDefinition


class RetryHandler:
    def __init__(self, prompt_engine, llm_router, output_parser, quality_evaluator, max_retry: int = 3) -> None:
        self._prompt_engine = prompt_engine
        self._llm_router = llm_router
        self._output_parser = output_parser
        self._quality_evaluator = quality_evaluator
        self._max_retry = max_retry

    async def run(
        self, request: GenerateRequest
    ) -> tuple[Optional[SchemaDefinition], Optional[QualityScore], int]:
        error_context: Optional[str] = None
        last_schema: Optional[SchemaDefinition] = None
        last_quality: Optional[QualityScore] = None

        for attempt in range(self._max_retry + 1):
            prompt = await self._prompt_engine.build_generate_prompt(
                description=request.description,
                target_db=request.target_db,
                conversation_history=request.conversation_history,
                error_context=error_context,
            )

            raw = await self._llm_router.generate(prompt)
            schema, parse_error = self._output_parser.parse(raw)

            if schema is None:
                error_context = f"Parse error on attempt {attempt + 1}:\n{parse_error}"
                continue

            last_schema = schema
            quality = await self._quality_evaluator.evaluate(schema, request.description)
            last_quality = quality

            if quality.passed:
                return schema, quality, attempt

            error_context = self._build_error_context(quality, attempt)

        return last_schema, last_quality, self._max_retry

    def _build_error_context(self, quality: QualityScore, attempt: int) -> str:
        lines = [f"Attempt {attempt + 1} quality score: {quality.composite:.2f} (threshold: 0.8)"]
        lines.append("Dimensions that need improvement:")

        if quality.syntax < 0.9:
            lines.append(f"- syntax ({quality.syntax:.2f}): Fix SQL syntax errors in DDL statements")
        if quality.integrity < 0.9:
            lines.append(
                f"- integrity ({quality.integrity:.2f}): "
                "Ensure every table has a primary key; verify all FK references point to existing tables/columns"
            )
        if quality.naming < 0.9:
            lines.append(
                f"- naming ({quality.naming:.2f}): "
                "Use snake_case for all table and column names; avoid SQL reserved words; keep names under 63 chars"
            )
        if quality.completeness < 0.9:
            lines.append(
                f"- completeness ({quality.completeness:.2f}): "
                "Ensure all entities from the description are represented; add missing relationships and attributes"
            )

        return "\n".join(lines)
