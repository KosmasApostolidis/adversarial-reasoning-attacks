#!/usr/bin/env bash
# Full benchmark sweep entrypoint.
# Phase 2 invocation: `./scripts/run_benchmark.sh configs/experiment.yaml`
# Phase 0/1 smoke:     `./scripts/run_benchmark.sh configs/smoke.yaml`

set -euo pipefail

CONFIG="${1:-configs/experiment.yaml}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[run_benchmark] config not found: ${CONFIG}" >&2
  exit 2
fi

echo "[run_benchmark] config=${CONFIG}"
python -m adversarial_reasoning.runner --config "${CONFIG}"
