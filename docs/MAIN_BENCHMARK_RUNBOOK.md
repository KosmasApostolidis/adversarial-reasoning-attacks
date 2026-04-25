# Main-benchmark runbook — clone to paper artifacts

End-to-end procedure to take a fresh checkout of this repo through every
step a paper submission needs: environment, data, weights, sanity gates,
the Phase-2 main_benchmark sweep, the (optional) Phase-3 transfer
evaluation, and the Phase-4 stats table + figure regeneration. Code is
already in `main` as of commit `c917f8e` (`docs: add MAIN_BENCHMARK_RUNBOOK.md`)
and the fan-out harness landed in `8756def` (`feat(phase2): main_benchmark
fan-out harness + config hygiene`).

The default target is the 3-model publication run
(Qwen2.5-VL-7B + LLaVA-Next-7B + Llama-3.2-11B-Vision, ≈ 40 GPU-days on a
free H200). A 2-model variant (Qwen + LLaVA, ≈ 13 GPU-days) is also
documented; pick before §0.3.

> Two known code blockers must be addressed before §4 produces correct
> outputs:
> 1. `scripts/build_stats_table.py` does not yet exist — required for §4.1.
> 2. `scripts/make_paper_figures.py` (and 3 sibling monoliths) load
>    Phase-1 paths (`runs/smoke/`, `runs/pgd_smoke/`, …) instead of
>    `runs/main/<mode>/` — required fix for §4.2.
>
> Both are flagged inline at the relevant step. The runbook tells you
> what the spec/fix looks like; the actual code change is out of scope
> for this document.

Run every command from the repo root unless noted otherwise.

```bash
cd /home/medadmin/kosmasapostolidis/adversarial-reasoning-attacks
```

---

## 0. Prerequisites (one-time, ~1–2 hours wall-time excluding TCIA download)

### 0.1 Hardware / disk / GPU

```bash
# GPU free?  the H200 is shared.  cancel/wait if another job is hogging it.
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# expected when free:  no compute-app rows under the header

# Disk free.  Phase-2 records are small (~30 MB); HF model cache is the
# big number.  3-model run footprint:
#   Qwen2.5-VL-7B  fp16  : ~14 GB
#   LLaVA-Next-7B  fp16  : ~14 GB
#   Llama-3.2-11B  fp16  : ~22 GB
#   ProstateX raw DICOM  : ~30 GB  (TCIA bulk download, §0.4)
#   ProstateX processed  : ~5  GB
#   Phase-2 runs/main    : ~30 MB
#   Total worst-case     : ~85 GB
df -h /home/medadmin | tail -2
# require: ≥100 GB free on the partition holding $HF_HOME for safety margin
```

### 0.2 Python environment

Two paths; pick one. Both produce the same dependency set per
`pyproject.toml` and `environment.yml`.

```bash
# Path A: conda (recommended — pinned versions, conda channel for CUDA torch)
conda env create -f environment.yml
conda activate adversarial-reasoning

# Path B: venv + pip (faster on machines without conda)
python3.11 -m venv .venv && source .venv/bin/activate
make install-dev          # pip install -e .[dev] + pre-commit install
```

### 0.3 HF_TOKEN + Llama-3.2 license  (skip if running 2-model variant)

The Llama-3.2-11B-Vision weights are gated. You need:

1. A HuggingFace account with the Meta Llama-3.2 license accepted at
   <https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct>.
2. An access token with read scope.

```bash
# Persist the token in the current shell (adjust ~/.bashrc if you want it
# permanent — keep it OUT of any repo file).
huggingface-cli login                    # paste token when prompted
# OR, scriptable:
export HF_TOKEN=hf_xxx_paste_here
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=./.hf_cache               # project-local cache (already symlinked)
```

`.env.example` lists every env var the runner consults; copy it to
`.env` and fill in if you prefer dotenv-style management.

### 0.4 Dataset acquisition

ProstateX (the primary dataset) is gated by the TCIA Data Use Agreement
and must be downloaded manually before the runbook can proceed past
this point.

```bash
bash scripts/prepare_datasets.sh
# Creates  data/prostatex/{raw,processed}, data/vqa_rad/, data/slake/.
# Downloads VQA-RAD and SLAKE from HuggingFace Hub automatically.
# ProstateX is NOT downloaded — see below.
```

ProstateX manual step:

1. Create a free TCIA account at <https://wiki.cancerimagingarchive.net/>.
2. Accept the ProstateX Data Use Agreement.
3. Install the NBIA Data Retriever (TCIA's bulk-download client).
4. Pull the bi-parametric T2/DWI/ADC subset into `data/prostatex/raw/`
   (~30 GB DICOM).

The runner reads from `data/prostatex/processed/` after preprocessing —
the processed split is created on-demand by the dataset loader on first
use; no separate batch step.

### 0.5 Model weight download

3-model variant (default — adjust `scripts/download_models.sh` first):

```bash
# Uncomment the Llama line in scripts/download_models.sh:
sed -i 's|^  # "meta-llama/Llama-3.2-11B-Vision-Instruct"|  "meta-llama/Llama-3.2-11B-Vision-Instruct"|' \
  scripts/download_models.sh

# Verify (should print one uncommented Llama line):
grep -n 'Llama-3.2' scripts/download_models.sh
```

```bash
HF_HOME=./.hf_cache HF_HUB_ENABLE_HF_TRANSFER=1 \
  bash scripts/download_models.sh
# pulls Qwen + LLaVA + Llama snapshots into ./.hf_cache/
# ~50 GB total transfer; expect 30–60 min on a typical research network
```

2-model variant: skip the `sed` step; just run the script.

### 0.6 Quality gate

```bash
make test    # pytest -q -m "not gpu and not slow"
# expected: 66 passed
make lint    # ruff check src/ scripts/ tests/  +  mypy src/
# expected: 0 errors, 0 warnings (2 pre-existing mypy notes are tolerated)
```

If either fails the rest of the runbook is invalid — fix or revert
before continuing. CI also runs the same gate on every push to `main`
(`.github/workflows/test.yml`, `.github/workflows/lint.yml`).

### 0.7 Smoke-load sanity

Before launching a multi-day sweep, verify each enabled VLM actually
loads on this GPU and produces a sensible forward pass. This catches
HF_TOKEN, license, processor-cache, and dtype/device misconfigurations
in ~30 s instead of 12 h into the sweep.

```bash
python - <<'PY'
from PIL import Image
import numpy as np
from adversarial_reasoning.models.loader import load_hf_vlm

# Run for every model you intend to sweep.  3-model default:
for name in ("qwen2_5_vl_7b", "llava_v1_6_mistral_7b", "llama_3_2_vision_11b"):
    print(f"[smoke-load] {name} ...", flush=True)
    vlm = load_hf_vlm(name)
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    out = vlm.generate(image=img, prompt="Describe this image in one word.", max_new_tokens=4)
    print(f"  ok  out={out!r}")
PY
```

Each model should print `ok  out=<some short text>`. Any
`OSError`, `GatedRepoError`, `OutOfMemoryError`, or
`NotImplementedError` here is fatal — fix before §1.

---

## 1. Phase-0 scientific gates  (~5 min)

Two scientific calibration gates that must pass before the main sweep is
interpretable. Both are exercised by the test suite under `tests/test_gates.py`,
which `make test` (§0.6) already ran. To re-run only the gates explicitly:

```bash
python -m pytest tests/test_gates.py -v
# expected: all gate tests pass
```

What the gates check:

- **`preprocessing_transfer`** (`src/adversarial_reasoning/gates/preprocessing_transfer.py`):
  An ε=16/255 random perturbation must survive a HF-fp16 → PNG → Ollama
  preprocessing round-trip with effective L∞ ≥ 2/255. If this fails,
  attacks at lower ε are masked by image preprocessing and the sweep is
  meaningless. This gate is also a Phase-3 prerequisite; failure for any
  model means transfer-eval would silently produce zero attack signal on
  the Ollama side.
- **`noise_floor`** (`src/adversarial_reasoning/gates/noise_floor.py`):
  Per-model T=0 trajectory variance across 5 seeds. Establishes the
  baseline above which an attack effect is real signal vs. decoder
  stochasticity. The convention is `attack_effect ≥ 2 × median(noise_floor)`
  to count.

Standalone CLI for the gate functions does not exist yet (the modules
expose `run_preprocessing_transfer()` and `run_noise_floor()` library
calls); pytest is the supported entry point. The README's
`python -m adversarial_reasoning.gates.preprocessing_transfer` line is
aspirational — track adding `__main__` blocks as future work.

---

## 2. Phase-2 main_benchmark sweep  (~13 GPU-days for 2 models, ~40 for 3)

The runner pins one attack algorithm per process via `--mode`. The
fan-out driver `scripts/run_main_benchmark.sh` invokes the runner once
per algorithm (`noise`, `pgd`, `apgd`, `trajectory_drift`,
`targeted_tool`), each with its own config (`configs/main_<mode>.yaml`)
that scopes `cfg.attacks` to a single matching algorithm so the ε loop
does not multiply work.

### 2.1 Enable all 3 models — Llama-3.2-Vision

The 5 per-mode configs each carry their own model list and currently
list only Qwen + LLaVA. To run the 3-model variant, **add** the Llama
entry to all five files:

```bash
# Insert `    - llama_3_2_vision_11b` immediately after the LLaVA line in
# every per-mode config:
for f in configs/main_{noise,pgd,apgd,trajectory_drift,targeted_tool}.yaml; do
  sed -i '/^    - llava_v1_6_mistral_7b$/a\    - llama_3_2_vision_11b' "$f"
done

# Verify — should print one match per config (5 lines total):
grep -n '^    - llama_3_2_vision_11b' configs/main_*.yaml
```

Caveats:

- **`configs/experiment.yaml` is informational only** — NOT consumed by
  `scripts/run_main_benchmark.sh`. Editing only that file leaves the
  sweep at 2 models. (The runbook prior to this revision pointed at line
  13 of that file; that instruction was wrong.)
- Budget: ~40 GPU-days continuous on a free H200 (≈ 3× the 2-model
  path).
- The Llama wrapper (`src/adversarial_reasoning/models/llama_vision.py`)
  was last touched in the Phase-1 internals refactor; rerun §0.7 with
  `llama_3_2_vision_11b` specifically before launching the full sweep.

Skip this entire section to run the 2-model variant.

### 2.2 Recommended: full sequential run

```bash
# Starts noise → pgd → apgd → trajectory_drift → targeted_tool, in that
# order.  Logs are tee'd to runs/main/_logs/<mode>_<utc-timestamp>.log
# nohup + & lets you log out without killing the run.
nohup scripts/run_main_benchmark.sh > runs/main/_logs/_driver.out 2>&1 &
echo $!  # remember the PID; this is what to kill if you need to abort.
```

Total elapsed: ~13 GPU-days continuous on a free H200 (2-model), ~40 with
all 3 models. Expect ~2× wall-time under contention. Tail any leg from
another shell:

```bash
tail -f runs/main/_logs/<mode>_*.log
```

### 2.3 Reduced sample-count variant

`configs/tasks.yaml` declares 50 samples per task on the `dev` split. To
cap to 10 samples per task (≈ 2.5-day total for 2 models, ≈ 8-day for 3),
edit each `configs/main_<mode>.yaml` and add a `task_overrides` block:

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

Then launch §2.2 as normal.

### 2.4 Per-mode (drip-feed) variant

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

### 2.5 Tunables  (env vars to `run_main_benchmark.sh`)

```bash
PGD_STEPS=100  scripts/run_main_benchmark.sh apgd                # 100-step inner loop
SPLIT=test     scripts/run_main_benchmark.sh                     # use test split
MAX_STEPS=12   scripts/run_main_benchmark.sh trajectory_drift    # longer agent trajectory
```

Defaults from `scripts/run_main_benchmark.sh`: `PGD_STEPS=20`,
`SPLIT=dev`, `MAX_STEPS=8`.

### 2.6 Verify Phase-2 done

```bash
# Expected line counts (2-model, 3 tasks, 5 seeds, 5 ε per attack):
# noise            : 750 lines  (50 samples × 3 tasks × 5 seeds, ε ignored)
# pgd / apgd / …   : 750 lines  per attack (3 tasks × 50 samples × 5 seeds)
#                    × 5 ε = 3750 if eps_loop is multiplicative
# Reality: each per-mode config restricts attacks to one algorithm and
# the runner appends 1 row per (sample, seed, ε) cell — see runner.py
# for the exact loop order.
wc -l runs/main/*/records.jsonl
# every per-mode summary.json should report records > 0:
cat runs/main/*/summary.json | python -m json.tool

# pytest must still be green
make test
```

### 2.7 Resumption / partial-output recovery

The runner appends to `runs/main/<mode>/records.jsonl`. Re-running a
mode without cleanup will produce duplicate rows; aggregation code
deduplicates on `(model, task, attack, ε, seed, sample_id)` but it is
safer to wipe and relaunch.

Safe recovery:

```bash
# Clean a single failed leg and relaunch only that mode:
rm -rf runs/main/<mode>/
scripts/run_main_benchmark.sh <mode>
```

There is no checkpointing inside a leg. A crash mid-leg loses progress
on that leg; legs that already finished are unaffected.

---

## 3. Phase-3 — transfer evaluation (HF fp16 → Ollama Q4)  [OPTIONAL]

**Status: deferred. The paper ships without this section.**
Scaffolding is present (`OllamaVLMClient`, `load_ollama_vlm` in
`src/adversarial_reasoning/models/loader.py`) but the runner has no
`--mode transfer` and the figure script `transfer_rate_matrix.png` is
not yet implemented. ≈ 3–4 days of orchestration code to add it.

If you decide to skip Phase-3 (default for the v1 paper):

- The §6 manifest row "Optional: transfer matrix" is left empty.
- The paper draft notes "transfer evaluation deferred to follow-up work"
  in §Limitations.

If you decide to add it later, the implementation spec lives outside
this runbook — see `docs/PROJECT_REPORT.md` §Transfer evaluation for the
design and `configs/experiment.yaml` (`transfer_evaluation:` block) for
the intended config surface. At a minimum you will need:

1. Persist adversarial pixels from Phase-2 (currently NOT persisted —
   would require ~10 GB of pixel storage), OR re-run attacks
   deterministically at transfer time using the recorded
   `(seed, sample_id, ε)` to reconstruct the same perturbation.
2. Add `--mode transfer` to `runner.py` that reads
   `runs/main/<mode>/records.jsonl`, runs the Q4 target via
   `OllamaVLMClient.generate_from_image_bytes`, and writes
   `runs/transfer/<source>_to_<target>/<mode>/records.jsonl`.
3. Add a `transfer-rate` figure script under `scripts/` that writes
   `paper/figures/transfer/transfer_rate_matrix.png`.

---

## 4. Phase-4 — paper artifacts

### 4.1 Stats table  ⚠ blocker — `scripts/build_stats_table.py` does not exist

This is a hard prerequisite for the paper main result table. Build the
script before running the next command. Implementation spec:

- **Inputs**: `runs/main/{noise,pgd,apgd,trajectory_drift,targeted_tool}/records.jsonl`
- **Per-cell aggregation**: `model × task × attack × ε`
- **Per-cell metric**: `edit_distance_norm` median + bootstrap CI
  (10 000 resamples, 0.95 confidence level).
- **Significance test**: Wilcoxon signed-rank pairing each
  (attack, ε) cell against the matched `noise` baseline (same model ×
  task × seed × sample).
- **Multiple-comparison correction**: Benjamini-Hochberg, q = 0.05.
- **Output**: `paper/tables/main_benchmark.tex` (booktabs format,
  ready to `\input{}` from the manuscript).
- **Helpers already present**: `src/adversarial_reasoning/metrics/stats.py`
  exposes `wilcoxon_signed_rank_pair`, `benjamini_hochberg`, and
  `bootstrap_ci` with the right signatures — wire them up in the new
  script.

Once built, run:

```bash
python scripts/build_stats_table.py \
    --runs-dir runs/main \
    --out paper/tables/main_benchmark.tex
```

Expected output: a single `.tex` file ~3–5 KB with one row per
`(model, task, attack, ε)` cell flagging q-values below 0.05 with `*`
and reporting median Δ-edit-distance with [CI low, CI high].

### 4.2 Figures  ⚠ blocker — figure scripts read Phase-1 paths

`scripts/make_paper_figures.py`, `scripts/make_hero_figures.py`, and
`scripts/make_attack_landscape.py` each call `load_records()` with hard-coded
Phase-1 paths (`runs/smoke/`, `runs/pgd_smoke/`, `runs/smoke_sweep/`,
`runs/apgd_sweep/`, `runs/targeted_tool_sweep/`,
`runs/trajectory_drift_sweep/`, etc). Running `make figures` after
Phase-2 will silently regenerate the paper PNGs from Phase-1 sample data
and overwrite `paper/figures/paper/fig{1..5}_*.png` with stale content.

> **Required fix before regenerating figures:** rewrite the
> `load_records(...)` calls in the four monoliths
> (`scripts/make_paper_figures.py`, `scripts/make_hero_figures.py`,
> `scripts/make_attack_landscape.py`, `scripts/make_comprehensive_figures.py`)
> so each attack mode reads from `runs/main/<mode>/records.jsonl` (or
> filters a single concatenated JSONL by the `attack` field). Until that
> patch lands, treat the existing PNGs in `paper/figures/paper/` as
> Phase-1 placeholders, not paper outputs.

After the path fix:

```bash
make figures
# = adreason-figures paper          → paper/figures/paper/fig{1..5}_*.png
#   adreason-figures hero           → paper/figures/hero/*.png
#   adreason-figures attack-landscape → paper/figures/attack_landscape/*.png
```

`adreason-figures` is the console-script defined in `pyproject.toml` and
dispatches to `scripts/cli.py`. Other useful subcommands:
`comprehensive`, `reasoning-flow`, `graph`, `compare`, `figures`.

### 4.3 LaTeX manuscript

`paper/main.tex` does not yet exist. ≈ 1–2 weeks of writing once §4.1
and §4.2 outputs are final. Required `\input{}` / `\includegraphics{}`
targets:

- `paper/figures/paper/fig{1..5}_*.png` — generated in §4.2
- `paper/tables/main_benchmark.tex` — generated in §4.1
- `paper/figures/transfer/transfer_rate_matrix.png` — Phase-3 output
  (skip if §3 was deferred)

The §1–§3 narrative backbone is `docs/PROJECT_REPORT.md` (already in
repo, ~20 KB). Pull §1.5 (related work) and §2 (method) directly; §3
(results) is the new content paired with §4 outputs.

---

## 5. Reproducibility statement

Information to paste verbatim into the paper's reproducibility section:

### 5.1 Seed pinning

```text
seeds = [0, 1, 2, 3, 4]   # configs/main_<mode>.yaml :: experiment.seeds
```

The runner draws 5 trajectories per (sample, ε) cell, one per seed. Both
the agent decoder (`temperature=0.0`, `oracle_seeds=3`) and the attack
inner loop (`torch.manual_seed`) are seeded deterministically.

### 5.2 Commit hash pin

Capture immediately before launching §2.2:

```bash
git rev-parse HEAD > runs/main/_commit.txt
git diff --stat HEAD          # should be empty for a clean run
```

Paste the output of `git rev-parse HEAD` into the paper §Reproducibility
("All Phase-2 results were generated at commit `<hash>`.").

### 5.3 Determinism caveats

- VLM decoders are stochastic at the kernel level (CUDA non-determinism
  on attention reductions). The same seed produces the same trajectory
  to within decode-noise, not bit-exact. This is documented in the
  Phase-1 noise-floor results (`tests/test_gates.py::test_noise_floor`).
- The Llama-3.2-Vision tool-use protocol is prompt-scaffolded (Meta's
  official tool-use contract is text-only). Expect higher intra-seed
  trajectory variance than Qwen2.5-VL — see `configs/models.yaml`
  notes for `llama_3_2_vision_11b`.
- HF processor versions matter: a `transformers` upgrade between
  Phase-2 and a re-run can shift per-cell metrics by ≥ 1 σ. Pin the
  exact version with `pip freeze > runs/main/_pip_freeze.txt` at run
  start.

---

## 6. Submission artifact manifest

Runbook is "done" when every required row exists with a non-stale
mtime. Each entry lists the section that produces it.

| Required | Artifact                       | Path                                                       | Source |
|----------|--------------------------------|-------------------------------------------------------------|--------|
| Yes      | Per-mode raw records           | `runs/main/<mode>/records.jsonl`                            | §2     |
| Yes      | Per-mode summary               | `runs/main/<mode>/summary.json`                             | §2     |
| Yes      | Driver logs                    | `runs/main/_logs/<mode>_<utc-ts>.log`                       | §2     |
| Yes      | Commit pin                     | `runs/main/_commit.txt`                                     | §5.2   |
| Yes      | Pip freeze                     | `runs/main/_pip_freeze.txt`                                 | §5.3   |
| Yes      | Stats table                    | `paper/tables/main_benchmark.tex`                           | §4.1   |
| Yes      | Main figures                   | `paper/figures/paper/fig{1..5}_*.png`                       | §4.2   |
| Yes      | Hero / landscape figures       | `paper/figures/{hero,attack_landscape}/*.png`               | §4.2   |
| Yes      | Quality-gate proof             | `make test && make lint` final exit 0                       | §0.6 / §2.6 |
| Yes      | LaTeX source                   | `paper/main.tex`                                            | §4.3   |
| Optional | Transfer matrix                | `paper/figures/transfer/transfer_rate_matrix.png`           | §3     |

---

## 7. Abort / cleanup

```bash
# stop the driver:
kill <PID-from-step-2.2>

# and any python child it spawned:
pkill -f 'adversarial_reasoning.runner'

# nuke partial run output (does NOT touch published Phase-1 sweeps):
rm -rf runs/main/<mode>/                                         # specific leg only
rm -rf runs/main/_logs/                                          # log archive only

# nuke everything Phase-2 (use with care — wipes all main_benchmark output):
rm -rf runs/main/
```

---

## 8. Where things live (one-page reference)

| Topic                         | Path                                                                |
|-------------------------------|---------------------------------------------------------------------|
| This runbook                  | `docs/MAIN_BENCHMARK_RUNBOOK.md`                                    |
| Project narrative / paper backbone | `docs/PROJECT_REPORT.md`                                       |
| Per-mode launch configs       | `configs/main_{noise,pgd,apgd,trajectory_drift,targeted_tool}.yaml` |
| Master experiment config (informational) | `configs/experiment.yaml`                               |
| Model registry                | `configs/models.yaml`                                               |
| Task scenario registry        | `configs/tasks.yaml`                                                |
| Attack hyperparameters        | `configs/attacks.yaml`                                              |
| Fan-out driver                | `scripts/run_main_benchmark.sh`                                     |
| Runner CLI                    | `python -m adversarial_reasoning.runner --help`                     |
| Console scripts               | `adreason`, `adreason-figures`, `adreason-compare` (`pyproject.toml`) |
| Output records                | `runs/main/<mode>/records.jsonl` + `runs/main/<mode>/summary.json`  |
| Driver logs                   | `runs/main/_logs/<mode>_<utc-ts>.log`                               |
| Statistics primitives         | `src/adversarial_reasoning/metrics/stats.py`                        |
| Trajectory edit-distance      | `src/adversarial_reasoning/metrics/trajectory.py`                   |
| Phase-0 gate library          | `src/adversarial_reasoning/gates/{preprocessing_transfer,noise_floor}.py` |
| Model wrappers                | `src/adversarial_reasoning/models/{qwen_vl,llava,llama_vision}.py`  |
| Test suite                    | `tests/test_*.py` (66 tests; 13 files)                              |
| CI workflows                  | `.github/workflows/{test,lint}.yml`                                 |
| Make targets                  | `make help` — `install-dev`, `test`, `lint`, `figures`, `clean`     |
| Conda env spec                | `environment.yml`                                                   |
| Env-var sample                | `.env.example`                                                      |
