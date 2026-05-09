"""Reasoning-flow figures entrypoint — dispatches to scripts/reasoning_flow/ package.

graph11_reasoning_paths.png   — per-patient side-by-side paths through tool space
graph12_alluvial.png          — alluvial stream: tool flow across steps (benign vs PGD)
graph13_transition_delta.png  — delta graph: edges added/removed by PGD
graph14_reasoning_strips.png  — horizontal strip comparison with divergence markers
graph15_multi_condition.png   — all 3 conditions (benign/noise/PGD) in one step×tool grid
"""

from __future__ import annotations

from scripts.reasoning_flow import (
    graph11_reasoning_paths,
    graph12_alluvial,
    graph13_transition_delta,
    graph14_reasoning_strips,
    graph15_multi_condition,
)
from scripts.reasoning_flow._common import OUT


def main() -> int:
    graph11_reasoning_paths()
    graph12_alluvial()
    graph13_transition_delta()
    graph14_reasoning_strips()
    graph15_multi_condition()
    print(f"\nAll figures → {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
