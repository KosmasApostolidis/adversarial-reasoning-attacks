#!/usr/bin/env bash
# Phase-2 main_benchmark fan-out driver.
#
# The runner pins one attack algorithm per process via --mode. We invoke it
# 5 times (noise, pgd, apgd, trajectory_drift, targeted_tool), each with a
# matching configs/main_<mode>.yaml that scopes cfg.attacks to the single
# algorithm so the eps loop doesn't multiply work.
#
# Usage:
#   scripts/run_main_benchmark.sh                       # all 5 modes, sequential
#   scripts/run_main_benchmark.sh pgd apgd              # only listed modes
#   PGD_STEPS=100 scripts/run_main_benchmark.sh apgd    # APGD with 100 inner steps
#   SPLIT=test scripts/run_main_benchmark.sh            # use test split (50 samples)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODES=("$@")
if [[ ${#MODES[@]} -eq 0 ]]; then
    MODES=(noise pgd apgd trajectory_drift targeted_tool)
fi

PGD_STEPS="${PGD_STEPS:-20}"
SPLIT="${SPLIT:-dev}"
MAX_STEPS="${MAX_STEPS:-8}"

TS=$(date -u +'%Y%m%dT%H%M%SZ')
LOG_DIR="runs/main/_logs"
mkdir -p "$LOG_DIR"

echo "[main_benchmark] modes=${MODES[*]}  split=$SPLIT  pgd_steps=$PGD_STEPS  ts=$TS"

for mode in "${MODES[@]}"; do
    CFG="configs/main_${mode}.yaml"
    if [[ ! -f "$CFG" ]]; then
        echo "[main_benchmark] SKIP $mode — $CFG not found" >&2
        continue
    fi
    LOG="$LOG_DIR/${mode}_${TS}.log"
    echo "[main_benchmark] >>> $mode  cfg=$CFG  log=$LOG"
    python -m adversarial_reasoning.runner \
        --config "$CFG" \
        --mode "$mode" \
        --split "$SPLIT" \
        --pgd-steps "$PGD_STEPS" \
        --max-steps "$MAX_STEPS" \
        2>&1 | tee "$LOG"
    echo "[main_benchmark] <<< $mode done"
done

echo "[main_benchmark] all modes complete  ts=$TS"
