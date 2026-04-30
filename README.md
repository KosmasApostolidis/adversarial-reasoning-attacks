# Adversarial Reasoning Attacks on Medical Imaging VLM Agents

White-box image-space adversarial perturbations against open-source Vision-Language Model (VLM) agents operating in a medical imaging context. We measure whether ε-bounded pixel-space perturbations induce systematic, reproducible deviations in agent tool-call trajectories, and whether such effects persist after transfer from HuggingFace fp16 surrogates to Ollama Q4-quantized deployment.

## Research question

Do ε-bounded pixel-space adversarial perturbations on medical images induce systematic, reproducible deviations in agent tool-call behaviour versus benign images — and does the effect persist after transfer from HF fp16 surrogate to Ollama Q4 quantized deployment?

## Targets

| Phase | VLM | Tool support |
|-------|-----|--------------|
| 0/1 | Qwen2.5-VL-7B-Instruct | native function calling |
| 0/1 | LLaVA-v1.6-Mistral-7B | prompt-scaffolded (ReAct) |
| 2 (deferred) | Llama-3.2-11B-Vision-Instruct | prompt-scaffolded only |

## Primary dataset

**ProstateX** (public, TCIA) — prostate MRI (T2 / DWI / ADC), bi-parametric subset with PI-RADS labels, 3-fold cross-validation split (163 patients). **ProstateX-2** (TCIA, DICOM) is fetched and preprocessed separately via `scripts/fetch_prostatex2_tcia.py` + `scripts/preprocess_prostatex2_dicom.py`. Cross-domain secondary: VQA-RAD, SLAKE.

## Attack suite

| Attack | Type | Loss |
|--------|------|------|
| PGD L∞ | untargeted baseline | CE on tool-call tokens |
| APGD L∞ | untargeted stronger | adaptive-step PGD |
| C&W L2 | targeted | margin-based |
| Trajectory-Drift PGD (custom) | untargeted | `-KL(p_attack ‖ p_benign)` over tool-name positions |
| Targeted-Tool PGD (custom) | targeted | CE forcing target tool at step k |

ε sweep: `{2, 4, 8, 16}/255` at L∞.

## Metrics

- Trajectory edit distance (Levenshtein on tool-name sequence)
- Tool-selection flip rate @ step k
- Targeted-tool hit rate
- Task success delta
- Param L1 distance on numeric tool args
- Attack transfer rate (Ollama Q4 / HF fp16)

Statistics: paired Wilcoxon signed-rank, bootstrap 95% CIs (10 000 resamples), Benjamini-Hochberg FDR at q=0.05.

## Quickstart

```bash
# 1. Environment
conda create -n advreasoning python=3.11 -y
conda activate advreasoning
pip install -e .[dev]

# 2. Model weights (Qwen2.5-VL-7B + LLaVA-v1.6-Mistral-7B)
bash scripts/download_models.sh

# 3. Data prep (ProstateX 3-fold CV splits required by pipeline)
bash scripts/prepare_datasets.sh

# 4. Sanity gates
python -m adversarial_reasoning.gates.preprocessing_transfer
python -m adversarial_reasoning.gates.noise_floor

# 5. Full benchmark sweep (3-fold CV × 5 attacks × 2 VLMs + stats + figures)
bash scripts/run_full_attack_pipeline.sh

# Subset runs — pass mode names as positional args
bash scripts/run_full_attack_pipeline.sh pgd apgd

# Skip runner, regenerate figures and stats table from existing records
SKIP_SWEEP=1 bash scripts/run_full_attack_pipeline.sh
```

See `docs/MAIN_BENCHMARK_RUNBOOK.md` for the full end-to-end procedure including GPU requirements, TCIA download instructions, and Phase-3 transfer evaluation.

## Output layout

```
runs/main/<mode>/<fold>/records.jsonl   # per (mode, fold) raw records
runs/main/<mode>/records.jsonl          # concatenated across folds
paper/tables/main_benchmark.tex         # stats table
paper/figures/hero/                     # hero panel
paper/figures/attack_landscape/         # attack landscape
paper/figures/cross_model/              # Qwen vs LLaVA
paper/figures/attack_comparison/        # all-attacks aggregate
paper/figures/graphs_v2/               # reasoning-flow graphs
```

## Reproducibility

All runs record: torch/transformers/torchattacks/ART versions, Ollama model digests, fixed seeds. Seeds are stable across Python hash-randomisation (`PYTHONHASHSEED`-independent). See `docs/reproducibility.md`.

## Ethics

**This benchmark is a defensive research tool.** Do not deploy in clinical settings. All attacks are sandboxed — no real-world clinical system is targeted. See `docs/ethics.md`.

## License

Code: MIT. Data sources carry their own licenses (ProstateX via TCIA DUA, VQA-RAD CC-BY, SLAKE CC-BY-SA). See `LICENSE`.

## Citation

See `CITATION.cff`.
