#!/usr/bin/env bash
# Full-pipeline driver: 3-fold ProstateX CV x 5 attack modes x 2 VLMs x 3 tasks,
# followed by stats table + every paper figure script.
#
# Output layout:
#   runs/main/<mode>/<fold>/records.jsonl       # per (mode, fold) records (raw)
#   runs/main/<mode>/records.jsonl              # concatenated across folds
#   runs/main/_logs/<mode>_<fold>_<utc>.log     # per-invocation log
#   runs/main/_logs/figure_<name>_<utc>.log     # per-figure log
#   paper/tables/main_benchmark.tex             # stats table
#   paper/figures/paper/                        # paper-ready composite figures
#   paper/figures/hero/                         # hero panel
#   paper/figures/attack_landscape/             # attack landscape
#   paper/figures/cross_model/                  # qwen vs llava
#   paper/figures/attack_comparison/            # all-attacks aggregate
#   paper/figures/pgd_vs_noise/                 # PGD vs noise box
#   paper/figures/stats/                        # comprehensive stats
#   paper/figures/graphs_v2/                    # comprehensive graphs
#   paper/figures/reasoning_flow/               # reasoning-flow figures
#   paper/figures/<model>_*                     # per-model make_figures.py outputs
#
# Folds are driven via runner --split flag, mapped through the
# task-config bhi_split_to_fold table (configs/tasks.yaml):
#   --split train -> fold_1   (55 patients)
#   --split dev   -> fold_2   (54 patients)
#   --split test  -> fold_3   (54 patients)
#
# Usage:
#   scripts/run_full_attack_pipeline.sh                        # full sweep + all figures
#   scripts/run_full_attack_pipeline.sh pgd apgd               # subset of modes
#   FOLDS="train dev" scripts/run_full_attack_pipeline.sh      # subset of folds
#   PGD_STEPS=100 scripts/run_full_attack_pipeline.sh apgd     # APGD with 100 inner steps
#   SKIP_SWEEP=1   scripts/run_full_attack_pipeline.sh         # skip runner, only post-process
#   SKIP_FIGURES=1 scripts/run_full_attack_pipeline.sh         # sweep only, no figures
#   SKIP_TABLE=1   scripts/run_full_attack_pipeline.sh         # no stats table
#
# Required env (only when running gradient/llava modes):
#   HF_TOKEN                  HuggingFace token (gated Llama-3.2-Vision).
#                             Not required for `noise`-only runs.
#
# Optional env:
#   AR_PROSTATEX_BHI_ROOT     ProstateX cv_folds path (default: data/prostatex/processed/cv_folds)
#   HF_HOME                   HF cache dir (default: $REPO_ROOT/.hf_cache)
#   PGD_STEPS                 gradient-attack inner steps (default: 20)
#   MAX_STEPS                 agent rollout horizon (default: 8)
#   FOLDS                     space-separated split names (default: "train dev test")
#   FIGURE_TASKS              tasks to plot trajectory figure for (default: all 3)
#   MODELS_FOR_PERMODEL_FIGS  (default: "qwen2_5_vl_7b llava_v1_6_mistral_7b")
#   CUDA_VISIBLE_DEVICES      GPU ids

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VALID_MODES=(noise pgd apgd trajectory_drift targeted_tool)
HF_GATED_MODES=(pgd apgd trajectory_drift targeted_tool)

is_in() {
    local needle="$1"; shift
    local x
    for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
    return 1
}

# --- knobs -------------------------------------------------------------
MODES=("$@")
if [[ ${#MODES[@]} -eq 0 ]]; then
    MODES=("${VALID_MODES[@]}")
fi

for mode in "${MODES[@]}"; do
    if ! is_in "$mode" "${VALID_MODES[@]}"; then
        echo "[full-pipeline] FATAL: unknown mode '$mode' (valid: ${VALID_MODES[*]})" >&2
        exit 2
    fi
done

# --- preflight ---------------------------------------------------------
needs_hf=0
for mode in "${MODES[@]}"; do
    if is_in "$mode" "${HF_GATED_MODES[@]}"; then needs_hf=1; break; fi
done
if [[ "$needs_hf" -eq 1 ]]; then
    : "${HF_TOKEN:?HF_TOKEN is required for gradient/targeted modes (gated Llama-3.2-Vision)}"
fi

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export AR_PROSTATEX_BHI_ROOT="${AR_PROSTATEX_BHI_ROOT:-$REPO_ROOT/data/prostatex/processed/cv_folds}"

for f in fold_1 fold_2 fold_3; do
    if [[ ! -d "$AR_PROSTATEX_BHI_ROOT/$f" ]]; then
        echo "[full-pipeline] FATAL: missing $AR_PROSTATEX_BHI_ROOT/$f" >&2
        echo "[full-pipeline] hint: run scripts/prepare_datasets.sh first" >&2
        exit 1
    fi
done

read -r -a SPLITS <<< "${FOLDS:-train dev test}"
read -r -a FIGURE_TASKS <<< "${FIGURE_TASKS:-prostate_mri_workup rad_vqa_action prostate_mri_targeted}"
read -r -a PERMODEL_MODELS <<< "${MODELS_FOR_PERMODEL_FIGS:-qwen2_5_vl_7b llava_v1_6_mistral_7b}"

PGD_STEPS="${PGD_STEPS:-20}"
MAX_STEPS="${MAX_STEPS:-8}"

TS=$(date -u +'%Y%m%dT%H%M%SZ')
RUN_ROOT="runs/main"
LOG_DIR="$RUN_ROOT/_logs"
FIG_ROOT="paper/figures"
TABLE_OUT="paper/tables/main_benchmark.tex"
mkdir -p "$LOG_DIR" "$FIG_ROOT/paper" "$(dirname "$TABLE_OUT")"

declare -A SPLIT_TO_FOLD=(
    [train]=fold_1
    [val]=fold_1   # alias: bhi_split_to_fold maps val -> fold_1 (held-out eval)
    [dev]=fold_2
    [test]=fold_3
)

warn() { echo "[full-pipeline] WARN: $*" >&2; }

run_step() {
    local label="$1"; shift
    local log="$LOG_DIR/figure_${label}_${TS}.log"
    local rc=0
    echo "[full-pipeline] >>> figure: $label"
    "$@" >"$log" 2>&1 || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        echo "[full-pipeline] <<< figure: $label  ok  (log: $log)"
    else
        warn "$label failed (rc=$rc); see $log"
    fi
}

# --- sweep loop --------------------------------------------------------
if [[ "${SKIP_SWEEP:-0}" == "1" ]]; then
    echo "[full-pipeline] SKIP_SWEEP=1; skipping runner sweep"
else
    echo "[full-pipeline] modes=${MODES[*]}  splits=${SPLITS[*]}  pgd_steps=$PGD_STEPS  max_steps=$MAX_STEPS  ts=$TS"

    for mode in "${MODES[@]}"; do
        CFG="configs/main_${mode}.yaml"
        if [[ ! -f "$CFG" ]]; then
            warn "SKIP $mode -- $CFG not found"
            continue
        fi

        for split in "${SPLITS[@]}"; do
            fold_tag="${SPLIT_TO_FOLD[$split]:-$split}"
            OUT_DIR="$RUN_ROOT/$mode/$fold_tag"
            LOG="$LOG_DIR/${mode}_${fold_tag}_${TS}.log"

            echo "[full-pipeline] >>> mode=$mode  split=$split  fold=$fold_tag  out=$OUT_DIR"
            mkdir -p "$OUT_DIR"

            if ! python -m adversarial_reasoning.runner \
                --config "$CFG" \
                --mode "$mode" \
                --split "$split" \
                --out "$OUT_DIR" \
                --pgd-steps "$PGD_STEPS" \
                --max-steps "$MAX_STEPS" \
                2>&1 | tee "$LOG"
            then
                warn "runner crashed for mode=$mode fold=$fold_tag (continuing)"
            fi

            echo "[full-pipeline] <<< mode=$mode  fold=$fold_tag  done"
        done
    done

    echo "[full-pipeline] sweep complete  ts=$TS"
fi

# --- concatenate per-fold jsonls into runs/main/<mode>/records.jsonl --
# Several legacy figure scripts (make_hero_figures, make_paper_figures,
# make_attack_landscape, make_comprehensive_figures, etc.) hard-code the
# path runs/main/<mode>/records.jsonl. Build that aggregate from the per-fold
# files so they Just Work.
echo "[full-pipeline] concatenating per-fold records.jsonl"
for mode_dir in "$RUN_ROOT"/*/; do
    mode=$(basename "$mode_dir")
    [[ "$mode" == "_logs" || "$mode" == "_per_model" ]] && continue
    out="$mode_dir/records.jsonl"
    shopt -s nullglob
    fold_files=("$mode_dir"/fold_*/records.jsonl)
    shopt -u nullglob
    if [[ ${#fold_files[@]} -eq 0 ]]; then
        warn "no per-fold jsonl under $mode_dir; keeping existing $out (if any)"
        continue
    fi
    : > "$out"
    for jf in "${fold_files[@]}"; do
        cat "$jf" >> "$out"
    done
    n=$(wc -l < "$out")
    echo "[full-pipeline] aggregated $out  ($n lines)"
done

if [[ "${SKIP_FIGURES:-0}" == "1" ]]; then
    echo "[full-pipeline] SKIP_FIGURES=1; figure step skipped"
    [[ "${SKIP_TABLE:-0}" == "1" ]] && exit 0
fi

# --- stats table -------------------------------------------------------
if [[ "${SKIP_TABLE:-0}" != "1" ]]; then
    run_step "build_stats_table" \
        python scripts/build_stats_table.py \
            --runs-dir "$RUN_ROOT" \
            --out "$TABLE_OUT"
fi

[[ "${SKIP_FIGURES:-0}" == "1" ]] && { echo "[full-pipeline] all done (no figures)"; exit 0; }

# --- trajectory length figure (one per task) --------------------------
for task in "${FIGURE_TASKS[@]}"; do
    fig_out="$FIG_ROOT/paper/fig_trajectory_length_before_after__${task}.png"
    run_step "traj_len_${task}" \
        python scripts/plot_trajectory_length_before_after.py \
            --runs-root "$RUN_ROOT" \
            --task "$task" \
            --out "$fig_out"
done

# --- per-(model,mode) make_figures.py ---------------------------------
# make_figures.py emits per-model files like edit_distance_distribution_<model>.png
# Drives off a single records.jsonl; we feed it the per-mode aggregate.
for mode in "${MODES[@]}"; do
    for task in "${FIGURE_TASKS[@]}"; do
        rec="$RUN_ROOT/$mode/records.jsonl"
        [[ -s "$rec" ]] || { warn "skip make_figures: empty $rec"; continue; }
        out_dir="$FIG_ROOT/per_mode/${mode}__${task}"
        mkdir -p "$out_dir"
        run_step "make_figures_${mode}_${task}" \
            python scripts/make_figures.py \
                --records "$rec" \
                --task "$task" \
                --out "$out_dir"
    done
done

# --- composite paper / hero / landscape / reasoning / comprehensive ---
# These read from runs/main/<mode>/records.jsonl directly (no CLI args).
run_step "make_paper_figures"        python scripts/make_paper_figures.py
run_step "make_hero_figures"         python scripts/make_hero_figures.py
run_step "make_attack_landscape"     python scripts/make_attack_landscape.py
run_step "make_reasoning_flow"       python scripts/make_reasoning_flow_figures.py
run_step "make_comprehensive"        python scripts/make_comprehensive_figures.py

# --- per-model jsonl splits (needed by make_compare_figures.py) -------
# make_compare_figures.py does NOT filter by model_key; it reads the full file.
# Pre-split per (mode, model) so qwen/llava boxplots are actually different.
SPLIT_ROOT="$RUN_ROOT/_per_model"
mkdir -p "$SPLIT_ROOT"

if ! command -v jq >/dev/null 2>&1; then
    warn "jq not found; per-model split skipped (install jq to enable cross-model figures)"
else
    for mode in "${MODES[@]}"; do
        rec="$RUN_ROOT/$mode/records.jsonl"
        [[ -s "$rec" ]] || continue
        for mkey in "${PERMODEL_MODELS[@]}"; do
            out_jsonl="$SPLIT_ROOT/${mode}__${mkey}.jsonl"
            if ! jq -c --arg mk "$mkey" 'select(.model_key == $mk)' "$rec" > "$out_jsonl"; then
                warn "split failed: $mode/$mkey"
            fi
        done
    done
fi

# --- cross-model comparison (qwen vs llava, on PGD records) -----------
QWEN_REC="$SPLIT_ROOT/pgd__qwen2_5_vl_7b.jsonl"
LLAVA_REC="$SPLIT_ROOT/pgd__llava_v1_6_mistral_7b.jsonl"
if [[ -s "$QWEN_REC" && -s "$LLAVA_REC" ]]; then
    run_step "make_compare_figures" \
        python scripts/make_compare_figures.py \
            --qwen "$QWEN_REC" \
            --llava "$LLAVA_REC" \
            --out "$FIG_ROOT/cross_model"
else
    warn "skip make_compare_figures: per-model split empty (qwen=$QWEN_REC llava=$LLAVA_REC)"
fi

# compare_models.py — runs/-rooted cross-model figure
run_step "compare_models" \
    python scripts/compare_models.py \
        --runs runs \
        --out "$FIG_ROOT/cross_model"

# --- compare_attacks (aggregate + pgd_vs_noise) -----------------------
# Default mode: --runs uses nargs="+" — pass ALL name=path entries after a
# single --runs flag (repeated --runs would overwrite, not append).
COMPARE_ENTRIES=()
for mode in "${MODES[@]}"; do
    rec="$RUN_ROOT/$mode/records.jsonl"
    [[ -s "$rec" ]] && COMPARE_ENTRIES+=("${mode}=${rec}")
done
if [[ ${#COMPARE_ENTRIES[@]} -gt 0 ]]; then
    run_step "compare_attacks_default" \
        python scripts/compare_attacks.py \
            --mode default \
            --runs "${COMPARE_ENTRIES[@]}" \
            --out "$FIG_ROOT/attack_comparison"
fi

if [[ -s "$RUN_ROOT/noise/records.jsonl" && -s "$RUN_ROOT/pgd/records.jsonl" ]]; then
    run_step "compare_attacks_pgd_noise" \
        python scripts/compare_attacks.py \
            --mode pgd_noise \
            --noise "$RUN_ROOT/noise/records.jsonl" \
            --pgd "$RUN_ROOT/pgd/records.jsonl" \
            --out "$FIG_ROOT/pgd_vs_noise"
fi

echo "[full-pipeline] all done  ts=$TS"
echo "[full-pipeline] table  : $TABLE_OUT"
echo "[full-pipeline] figures: $FIG_ROOT/{paper,hero,attack_landscape,cross_model,attack_comparison,pgd_vs_noise,stats,graphs_v2,reasoning_flow,per_mode}"
