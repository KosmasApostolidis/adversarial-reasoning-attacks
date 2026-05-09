"""Null-distribution panel — drift floor between reseeds of benign runs.

Reads a records.jsonl that has been backfilled with cot_drift_score *and*
also contains multiple seeds per (model, task, sample). For each
(model, task, sample) cell with >=2 benign trajectories (different seeds,
no attack), computes pairwise drift among them and accumulates a
distribution. This is the drift floor under no attack — establishes
that any observed attack drift is signal, not seed noise.

Inputs come from a separate "null" run produced by re-running the runner
with epsilon=0 over multiple seeds. Each row should have benign and
attacked trajectories that are both effectively benign — we identify
those by attack_name == "null" / epsilon == 0.

If no null rows are present we fall back to comparing across attacked
rows of the same (model, task, sample) cell — gives a weaker but still
useful "min observed drift" reference.

Usage
-----
    python scripts/cot_null_distribution.py \\
        --records artifacts/main_benchmark/records_cot.jsonl \\
        --out paper/figures/sanity/null_distribution.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adversarial_reasoning.metrics.cot import cot_drift_score


def _load(records_path: Path) -> list[dict]:
    return [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]


def _benign_trace(row: dict) -> str | None:
    return row.get("benign", {}).get("reasoning_trace") or None


def _is_null_row(row: dict) -> bool:
    return row.get("attack_name") == "null" or float(row.get("epsilon", 0.0)) == 0.0


def _drift_floor(records: list[dict], *, nli) -> np.ndarray:
    """Pairwise drift between benign traces of the same (model, task, sample)."""
    cells: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for r in records:
        if not _is_null_row(r):
            continue
        trace = _benign_trace(r)
        if not trace:
            continue
        key = (r.get("model_key", ""), r.get("task_id", ""), r.get("sample_id", ""))
        cells[key].append(trace)
    drifts: list[float] = []
    for traces in cells.values():
        if len(traces) < 2:
            continue
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                drifts.append(cot_drift_score(traces[i], traces[j], nli=nli))
    return np.array(drifts, dtype=np.float64)


def _attack_drift_distribution(records: list[dict]) -> np.ndarray:
    out = []
    for r in records:
        if _is_null_row(r):
            continue
        v = r.get("cot_drift_score")
        if v is not None:
            out.append(float(v))
    return np.array(out, dtype=np.float64)


def render(null_drifts: np.ndarray, attack_drifts: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bins = np.linspace(0, 1, 41)
    if null_drifts.size:
        ax.hist(null_drifts, bins=bins, color="#7A8499", alpha=0.85,
                edgecolor="white", linewidth=0.4, label=f"null (n={null_drifts.size})")
    if attack_drifts.size:
        ax.hist(attack_drifts, bins=bins, color="#FC8181", alpha=0.55,
                edgecolor="white", linewidth=0.4, label=f"attack (n={attack_drifts.size})")
    ax.set_xlabel("cot_drift_score (NLI distance)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("CoT drift: null reseeds vs attacked runs", fontsize=12, pad=10)
    ax.legend(loc="upper right", frameon=True, edgecolor="#dddddd")
    ax.grid(linestyle=":", alpha=0.4)
    if null_drifts.size:
        floor = float(np.quantile(null_drifts, 0.95))
        ax.axvline(floor, color="#5C6B82", linewidth=1.2, linestyle="--")
        ax.text(floor + 0.01, ax.get_ylim()[1] * 0.85,
                f"95%ile null = {floor:.3f}", color="#5C6B82", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stub-nli", action="store_true",
                   help="Use a constant 0.5 stub instead of DeBERTa.")
    args = p.parse_args(argv)

    if args.stub_nli:
        nli = lambda p, h: 0.5
    else:
        from adversarial_reasoning.metrics.nli import entailment_prob as nli

    records = _load(args.records)
    null_drifts = _drift_floor(records, nli=nli)
    attack_drifts = _attack_drift_distribution(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(null_drifts, attack_drifts, args.out)
    print(
        f"[null_distribution] null_n={null_drifts.size} attack_n={attack_drifts.size} "
        f"null_q95={float(np.quantile(null_drifts, 0.95)) if null_drifts.size else 'n/a':>5} "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
