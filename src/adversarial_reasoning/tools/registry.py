"""Tool registry shared by agent loops and attack targeting code."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: Callable[..., Any]

    def schema_dict(self) -> dict[str, Any]:
        """OpenAI/JSON-schema-style representation for native-tool-calling models."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


class ToolRegistry:
    """Small dict-backed registry. Order preserved so JSON schemas are stable."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not in registry. Known: {list(self._tools)}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema_dict() for t in self._tools.values()]

    def subset(self, names: list[str]) -> ToolRegistry:
        new = ToolRegistry()
        for n in names:
            new.register(self.get(n))
        return new


def default_registry() -> ToolRegistry:
    """Build the 6-tool sandboxed registry used by the medical agent."""
    from . import (
        guidelines_db,
        pubmed_stub,
        report_drafter,
        risk_scores,
        workflow_actions,
    )

    registry = ToolRegistry()
    registry.register(guidelines_db.tool())
    registry.register(pubmed_stub.tool())
    registry.register(risk_scores.tool())
    registry.register(report_drafter.tool())
    registry.register(workflow_actions.request_followup_tool())
    registry.register(workflow_actions.escalate_tool())
    return registry
