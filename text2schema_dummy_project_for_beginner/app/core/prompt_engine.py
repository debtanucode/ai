from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml
from jinja2 import Environment, FileSystemLoader

from app.core.cache_manager import CacheManager
from app.models.schema import ConversationTurn, SchemaDefinition, TargetDB

_ROOT = Path(__file__).parent.parent.parent
_KNOWLEDGE_DIR = _ROOT / "knowledge"
_TEMPLATES_DIR = _ROOT / "templates"


class PromptEngine:
    def __init__(self, cache: CacheManager) -> None:
        self._cache = cache
        self._jinja = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=False,
        )
        self._output_schema = json.dumps(SchemaDefinition.model_json_schema(), indent=2)

    def _load_knowledge(self, target_db: TargetDB) -> dict:
        path = _KNOWLEDGE_DIR / target_db.value / "conventions.yaml"
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _format_context(self, knowledge: dict) -> str:
        parts: list[str] = []
        if conventions := knowledge.get("conventions"):
            parts.append("### Conventions")
            for c in conventions:
                parts.append(f"- {c}")
        if naming := knowledge.get("naming"):
            parts.append("\n### Naming Rules")
            for k, v in naming.items():
                parts.append(f"- {k}: {v}")
        if type_mappings := knowledge.get("type_mappings"):
            parts.append("\n### Type Mappings")
            for k, v in type_mappings.items():
                parts.append(f"- {k}: {v}")
        if examples := knowledge.get("few_shot_examples"):
            parts.append("\n### Examples")
            for ex in examples:
                parts.append(f"Input: {ex.get('input', '')}")
                parts.append(f"Output: {ex.get('output_summary', '')}")
        return "\n".join(parts)

    async def _get_domain_context(self, target_db: TargetDB) -> str:
        path = _KNOWLEDGE_DIR / target_db.value / "conventions.yaml"
        if not path.exists():
            return ""
        raw_content = path.read_text()
        cached = await self._cache.get_context(raw_content)
        if cached:
            return cached
        knowledge = yaml.safe_load(raw_content) or {}
        context = self._format_context(knowledge)
        await self._cache.set_context(raw_content, context)
        return context

    async def build_generate_prompt(
        self,
        description: str,
        target_db: TargetDB,
        conversation_history: Optional[list[ConversationTurn]] = None,
        error_context: Optional[str] = None,
    ) -> str:
        knowledge = self._load_knowledge(target_db)
        naming = knowledge.get("naming", {})
        domain_context = await self._get_domain_context(target_db)

        template = self._jinja.get_template("generate.j2")
        return template.render(
            description=description,
            target_db=target_db.value,
            domain_context=domain_context,
            output_schema=self._output_schema,
            naming_style=naming.get("table_style", "snake_case"),
            column_style=naming.get("column_style", "snake_case"),
            conversation_history=conversation_history or [],
            error_context=error_context or "",
        )

    async def build_judge_prompt(self, description: str, schema_json: str) -> str:
        template = self._jinja.get_template("judge.j2")
        return template.render(description=description, schema_json=schema_json)
