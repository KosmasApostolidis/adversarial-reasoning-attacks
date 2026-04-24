#!/usr/bin/env bash
# One-shot reproduction of the paper pipeline.
# Runs: gates → smoke → full sweep → figures → stats tables.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[reproduce] 1/4 — Phase 0 gates"
python -m adversarial_reasoning.gates.preprocessing_transfer || true
python -m adversarial_reasoning.gates.noise_floor || true

echo "[reproduce] 2/4 — Phase 0 smoke run"
bash scripts/run_benchmark.sh configs/smoke.yaml

echo "[reproduce] 3/4 — Full benchmark"
bash scripts/run_benchmark.sh configs/experiment.yaml

echo "[reproduce] 4/4 — Figures + tables (manual: open notebooks/)"
echo "[reproduce] done."
