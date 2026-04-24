"""Base agent abstractions + trajectory data structures.

Trajectories are the primary measurement target of this benchmark, so the
data schema here is load-bearing: every attack eval ultimately compares
two `Trajectory` instances (benign vs attacked) with the trajectory
metrics module.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation made during an agent's reasoning trace."""

    step: int
    name: str
    args: dict[str, Any]
    result: Any | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    """The full reasoning trace produced by one agent run.

    Fields:
        task_id: scenario instance id (matches `tasks/<task_id>/`)
        model_id: VLM identifier
        seed: RNG seed used for this run
        tool_calls: ordered list of ToolCall
        final_answer: free-text conclusion after tool calls complete
        reasoning_trace: raw text emitted between/around tool calls (CoT)
        metadata: extra fields (ε budget, attack name, etc.)
    """

    task_id: str
    model_id: str
    seed: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""
    reasoning_trace: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def tool_sequence(self) -> list[str]:
        """Ordered list of tool names — primary input to edit-distance metric."""
        return [c.name for c in self.tool_calls]

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "task_id": self.task_id,
                "model_id": self.model_id,
                "seed": self.seed,
                "tool_calls": [c.to_dict() for c in self.tool_calls],
                "final_answer": self.final_answer,
                "reasoning_trace": self.reasoning_trace,
                "metadata": self.metadata,
            },
            ensure_ascii=False,
        )


class AgentBase(ABC):
    """Abstract agent. Concrete impls wrap a VLM with a ReAct-style loop."""

    @abstractmethod
    def run(
        self,
        task_id: str,
        image: Any,
        prompt: str,
        *,
        seed: int = 0,
        max_steps: int = 8,
    ) -> Trajectory:
        """Execute the agent loop and return a complete Trajectory."""
