# Phase-2 main_benchmark — Runbook

Step-by-step instructions you run yourself to execute the remaining
research-pipeline phases. Code is already in `main` as of commit `8756def`
(`feat(phase2): main_benchmark fan-out harness + config hygiene`); this
document only describes **what to run, when, and how to verify**.

---

## 0. Prerequisites (5 min)

Run these in the repo root (`/home/medadmin/kosmasapostolidis/adversarial-reasoning-attacks`).

```bash
# 0.1 Test gate — must be green before launching anything compute-heavy
python -m pytest tests/ -q -m 'not gpu and not slow'
# expected: 66 passed
```

```bash
# 0.2 GPU free?  the H200 is shared.  cancel/wait if another job is hogging it.
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# expected when free:  no compute-app rows under the header
```

```bash
# 0.3 Disk free.  worst-case Phase-2 footprint is ~30 MB of records.jsonl
#                 (per-cell rows are ~3 KB; 1500 cells × 5 modes ≈ 22 MB).
df -h /home/medadmin | tail -2
# require: >2 GB free for safety margin (logs + Ollama models if you do Phase 3)
```

```bash
# 0.4 (Phase-2 with Llama-3.2-Vision ONLY)  HF_TOKEN must be set.
#     Skip if you are running Qwen + LLaVA only — that path needs no token.
huggingface-cli login                              # accept Meta's Llama-3.2 license first
#                                                    https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
#     Then re-enable the model in configs/experiment.yaml line 13 (uncomment).
```

---

## 1. Phase-2 launch — the full main_benchmark sweep (~13 GPU-days, 2 models)

The runner pins one attack algorithm per process via `--mode`. We fan out
across the 5 algorithms (`noise`, `pgd`, `apgd`, `trajectory_drift`,
`targeted_tool`), each with its own config (`configs/main_<mode>.yaml`)
that scopes `cfg.attacks` to a single matching algorithm so the eps loop
does not multiply work.

### 1A. Recommended: one shell invocation, all 5 modes sequentially

```bash
# starts noise → pgd → apgd → trajectory_drift → targeted_tool, in that order.
# logs are tee'd to runs/main/_logs/<mode>_<utc-timestamp>.log
#
# nohup + & lets you log out without killing the run.
# tail the log file from another shell to monitor progress.
nohup scripts/run_main_benchmark.sh > runs/main/_logs/_driver.out 2>&1 &
echo $!  # remember the PID; that's what to kill if you need to abort.
```

Total elapsed: ~13 days continuous on a free H200, ~2× that with GPU
contention. You can `tail -f runs/main/_logs/<mode>_*.log` from another
session at any time.

### 1B. Cheaper: reduce sample count

`configs/tasks.yaml` declares 50 samples per task on the `dev` split. To
cap to 10 samples per task (≈ 2.5-day total), edit each
`configs/main_<mode>.yaml` and add a `task_overrides` block, e.g.:

```yaml
experiment:
  ...
  tasks:
    - prostate_mri_workup
    - rad_vqa_action
    - prostate_mri_targeted
  task_overrides:                    # ← add this block
    prostate_mri_workup:
      dataset_split: { dev: 10 }
    rad_vqa_action:
      dataset_split: { dev: 10 }
    prostate_mri_targeted:
      dataset_split: { dev: 10 }
```

Then launch 1A as normal.

### 1C. Per-mode (drip-feed)

Run one algorithm at a time so you can inspect output between legs:

```bash
scripts/run_main_benchmark.sh pgd
# inspect:
ls -lh runs/main/pgd/records.jsonl runs/main/pgd/summary.json
head -1 runs/main/pgd/records.jsonl | python -m json.tool

# only continue once the above looks sane
scripts/run_main_benchmark.sh apgd
scripts/run_main_benchmark.sh trajectory_drift
scripts/run_main_benchmark.sh targeted_tool
scripts/run_main_benchmark.sh noise
```

### 1D. Tunables (all are env vars to `run_main_benchmark.sh`)

```bash
PGD_STEPS=100  scripts/run_main_benchmark.sh apgd                # 100-step inner loop
SPLIT=test     scripts/run_main_benchmark.sh                     # use test split
MAX_STEPS=12   scripts/run_main_benchmark.sh trajectory_drift    # longer agent trajectory
```

### 1E. Verify Phase 2 done

```bash
# 1500 cells × 5 modes ≈ 7500 lines total expected (qwen+llava only).
wc -l runs/main/*/records.jsonl
# every per-mode summary.json should report records > 0
cat runs/main/*/summary.json | python -m json.tool
# pytest must still be green
python -m pytest tests/ -q -m 'not gpu and not slow'
```

---

## 2. Phase-3 launch — transfer evaluation (HF fp16 → Ollama Q4)

Status: scaffolding present (`OllamaVLMClient`, `load_ollama_vlm`); the
runner has no `--mode transfer` yet. Implementation effort ≈ 3–4 days
of orchestration code, listed here so you can decide whether to do
it now or after Phase 4.

### 2A. Prerequisites

```bash
# Ollama daemon up
ollama serve &                                                   # if not already running
# Pull the Q4 targets defined in configs/models.yaml
ollama pull qwen2.5vl:7b-q4_K_M
ollama pull llava:7b-v1.6-mistral-q4_K_M
# (optional) ollama pull llama3.2-vision:11b-instruct-q4_K_M    # gated, needs Meta license
```

### 2B. What to build (work item, not a runnable command)

Add a `--mode transfer` to `src/adversarial_reasoning/runner.py` that:

1. Reads `runs/main/<mode>/records.jsonl` (Phase-2 outputs).
2. For each row, recovers the adversarial pixel tensor — currently NOT
   persisted; will require Phase 2 to be re-run with a `--save-pixels`
   flag, OR re-running the attack at inference time using the recorded
   `(seed, sample_id, eps)` to reconstruct the same perturbation.
3. Runs the Q4 target through `OllamaVLMClient.generate_from_image_bytes`.
4. Computes `trajectory_edit_distance(source_attacked, q4_attacked)`
   and writes `runs/transfer/<source>_to_<target>/<mode>/records.jsonl`.

Recommended approach: re-run attacks deterministically rather than
persisting pixels (~22 MB) × seeds × samples (~10 GB) of pixel storage.

### 2C. Verify Phase 3 done

```bash
ls runs/transfer/*/<mode>/records.jsonl
# transfer-rate figure (script also pending) renders to paper/figures/transfer/
```

---

## 3. Phase-4 launch — paper artifacts

### 3A. Stats table (1 day to write the script, then trivial to run)

Aggregate `runs/main/*/records.jsonl` into a per-cell table with
Wilcoxon p, BH-corrected q (q=0.05), and 10 000-resample bootstrap CI:

```bash
# script does not yet exist.  build it under scripts/build_stats_table.py
# using src/adversarial_reasoning/metrics/stats.py helpers.
python scripts/build_stats_table.py \
    --runs-dir runs/main \
    --out paper/tables/main_benchmark.tex
```

### 3B. Figure regen (after Phase 2 is fully done)

```bash
adreason-figures paper                                           # all 5 paper figures
adreason-figures hero                                            # 5 hero figures
adreason-figures attack-landscape                                # 5 landscape figures
# do NOT run before Phase 2 completes — outputs will be stale
```

### 3C. LaTeX manuscript

`paper/main.tex` does not yet exist. ≈ 1–2 weeks of writing once Phases
2 + 3 outputs are final. Pull in:

- `paper/figures/paper/fig{1..5}_*.png`
- `paper/tables/main_benchmark.tex`
- `paper/figures/transfer/transfer_rate_matrix.png` (Phase 3 output)
- `docs/PROJECT_REPORT.md` content as the §1–§3 backbone

---

## 4. Abort / cleanup

```bash
# stop the driver:
kill <PID-from-step-1A>
# and any python child it spawned:
pkill -f 'adversarial_reasoning.runner'
# nuke partial run output (does NOT touch published Phase-1 sweeps):
rm -rf runs/main/<mode>/                                         # specific leg only
```

---

## 5. Where things live (one-page reference)

| Artifact                         | Path                                                               |
|----------------------------------|--------------------------------------------------------------------|
| Per-mode launch configs          | `configs/main_{noise,pgd,apgd,trajectory_drift,targeted_tool}.yaml` |
| Fan-out driver                   | `scripts/run_main_benchmark.sh`                                    |
| Runner CLI                       | `python -m adversarial_reasoning.runner --help`                    |
| Output records                   | `runs/main/<mode>/records.jsonl` + `runs/main/<mode>/summary.json` |
| Logs                             | `runs/main/_logs/<mode>_<utc-timestamp>.log`                       |
| Statistics primitives            | `src/adversarial_reasoning/metrics/stats.py`                       |
| Trajectory edit-distance         | `src/adversarial_reasoning/metrics/trajectory.py`                  |
| Project report (paper backbone)  | `docs/PROJECT_REPORT.md`                                           |
| Earlier roadmap (this plan)      | `~/.claude/plans/refactor-the-whole-project-sequential-dolphin.md` |
