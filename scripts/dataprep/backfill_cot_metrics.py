"""Backfill CoT metrics onto an existing records.jsonl.

Reads a records.jsonl produced by ``python -m adversarial_reasoning.runner``,
recomputes the 7 CoT-metric fields (drift / faithfulness x{benign,attacked}
/ hallucination x{benign,attacked} / refusal x{benign,attacked}) plus 2
raw refusal probabilities for each row that has both benign and attacked
``reasoning_trace`` populated, and writes a new JSONL.

Idempotent: rows that already carry every CoT field are skipped (their
existing scores pass through unchanged). Rows missing reasoning_trace
(pre-v0.4.0 records) pass through unchanged with a warning logged once.

Usage
-----
    python scripts/backfill_cot_metrics.py \
        --in artifacts/main_benchmark/records.jsonl \
        --out artifacts/main_benchmark/records_cot.jsonl

The DeBERTa-v3-large-MNLI judge is loaded eagerly when this script
imports ``adversarial_reasoning.metrics.nli`` -- expect a one-time cold
start of ~30s on first run while the model downloads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from adversarial_reasoning.metrics.cot import score_pair

CoT_FIELDS = (
    "cot_drift_score",
    "cot_faithfulness_benign",
    "cot_faithfulness_attacked",
    "cot_hallucination_benign",
    "cot_hallucination_attacked",
    "cot_refusal_benign",
    "cot_refusal_attacked",
)


def _has_all_cot_fields(row: dict[str, Any]) -> bool:
    return all(k in row for k in CoT_FIELDS)


def _row_has_reasoning(row: dict[str, Any]) -> bool:
    benign = row.get("benign", {})
    attacked = row.get("attacked", {})
    return bool(benign.get("reasoning_trace")) and bool(attacked.get("reasoning_trace"))


def backfill(in_path: Path, out_path: Path, *, nli) -> dict[str, int]:
    counts = {"total": 0, "scored": 0, "skipped_existing": 0, "skipped_no_trace": 0}
    warned_no_trace = False
    with in_path.open() as f_in, out_path.open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            counts["total"] += 1
            row = json.loads(line)

            if _has_all_cot_fields(row):
                counts["skipped_existing"] += 1
                f_out.write(json.dumps(row, default=str) + "\n")
                continue

            if not _row_has_reasoning(row):
                counts["skipped_no_trace"] += 1
                if not warned_no_trace:
                    print(
                        f"[backfill] WARN: row missing reasoning_trace "
                        f"(model={row.get('model_key')}, task={row.get('task_id')}, "
                        f"sample={row.get('sample_id')}); passing through unchanged. "
                        f"(further such warnings suppressed)",
                        file=sys.stderr,
                    )
                    warned_no_trace = True
                f_out.write(json.dumps(row, default=str) + "\n")
                continue

            benign = row["benign"]
            attacked = row["attacked"]
            cot = score_pair(
                benign_cot=benign["reasoning_trace"],
                attacked_cot=attacked["reasoning_trace"],
                benign_tool_calls=benign.get("tool_calls", []),
                attacked_tool_calls=attacked.get("tool_calls", []),
                nli=nli,
            )
            row.update(cot)
            counts["scored"] += 1
            f_out.write(json.dumps(row, default=str) + "\n")
    return counts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    p.add_argument(
        "--stub-nli",
        action="store_true",
        help="Use a constant 0.5 stub instead of DeBERTa (testing/CI).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.stub_nli:

        def nli(p: str, h: str) -> float:
            return 0.5
    else:
        from adversarial_reasoning.metrics.nli import entailment_prob as nli
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = backfill(args.in_path, args.out_path, nli=nli)
    print(
        f"[backfill] in={args.in_path} out={args.out_path} "
        f"total={counts['total']} scored={counts['scored']} "
        f"skipped_existing={counts['skipped_existing']} "
        f"skipped_no_trace={counts['skipped_no_trace']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
