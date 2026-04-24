# Reproducibility

## Pinned dependencies

See `pyproject.toml`. Key pins:

- `torch>=2.6.0`
- `transformers>=4.45`
- `adversarial-robustness-toolbox>=1.20` (APGD / AutoAttack)
- `smolagents>=1.4`
- `ollama>=0.4`

(`torchattacks` was dropped — its 3.5.x wheels pin `requests~=2.25.1`, which
is irreconcilable with modern `datasets` / `smolagents`. Custom PGD in
`src/adversarial_reasoning/attacks/pgd.py` replaces the baseline.)

## Seeds

Every experimental cell is run with seeds `[0, 1, 2, 3, 4]`. Seeds control:

- `torch.manual_seed()` for VLM generation
- `numpy.random.default_rng()` for bootstrap CI
- PGD random-restart initialisation

## Run manifest

Each run writes a manifest JSON containing:

- Python version, torch / transformers / torchattacks / ART versions
- CUDA runtime version, GPU name, driver
- Commit SHA of the benchmark repo at run time
- Full config YAML(s)
- Ollama model digests (via `ollama list`) for every tag used
- Timestamps (start, per-cell, end)

## Data

- **ProstateX**: fetched via TCIA API. The exact subject list is frozen in
  `data/prostatex/manifest.csv` (committed).
- **VQA-RAD / SLAKE**: versioned HuggingFace snapshots are pinned by
  dataset revision.

## How to reproduce the paper

```bash
# 1. Environment
conda create -n advreasoning python=3.11 -y
conda activate advreasoning
pip install -e .[dev]

# 2. Weights + data
bash scripts/download_models.sh
bash scripts/prepare_datasets.sh

# 3. Phase 0 gates (must pass before any attack run)
python -m adversarial_reasoning.gates.preprocessing_transfer
python -m adversarial_reasoning.gates.noise_floor

# 4. Full benchmark
bash scripts/reproduce_paper.sh
```

Expected wall-clock time on an H200 GPU:
- Phase 0 (2 VLMs, smoke + gates): ~2 hours
- Phase 2 (3 VLMs, full sweep): re-estimated after Phase 0; upper bound ~4-6 days
