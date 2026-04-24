# Adversarial Reasoning Attacks on Medical Imaging VLM Agents

White-box image-space adversarial perturbations against open-source Vision-Language Model (VLM) agents operating in a medical imaging context. We measure whether ε-bounded pixel-space perturbations induce systematic, reproducible deviations in agent tool-call trajectories, and whether such effects persist after transfer from HuggingFace fp16 surrogates to Ollama Q4-quantized deployment.

## Research question

Do ε-bounded pixel-space adversarial perturbations on medical images induce systematic, reproducible deviations in agent tool-call behaviour versus benign images — and does the effect persist after transfer from HF fp16 surrogate to Ollama Q4 quantized deployment?

## Targets (Phase 0/1 → Phase 2)

| Phase | VLM | Tool support |
|-------|-----|--------------|
| 0/1 | Qwen2.5-VL-7B-Instruct | native function calling |
| 0/1 | LLaVA-v1.6-Mistral-7B | prompt-scaffolded (ReAct) |
| 2 (deferred) | Llama-3.2-11B-Vision-Instruct | prompt-scaffolded only (vision-mode) |

## Primary dataset

**ProstateX** (public, TCIA) — prostate MRI (T2 / DWI / ADC), bi-parametric subset with PI-RADS labels. Cross-domain secondary: VQA-RAD, SLAKE.

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

Statistics: paired Wilcoxon signed-rank, bootstrap 95% CIs (10000 resamples), Benjamini-Hochberg FDR at q=0.05.

## Quickstart

```bash
# 1. Environment
conda create -n advreasoning python=3.11 -y
conda activate advreasoning
pip install -e .[dev]

# 2. Model weights (Phase 0/1 only — 2 VLMs)
bash scripts/download_models.sh

# 3. Smoke test
pytest tests/test_models.py

# 4. Phase 0 gates
python -m adversarial_reasoning.gates.preprocessing_transfer
python -m adversarial_reasoning.gates.noise_floor

# 5. MVP signal check (Phase 1)
python -m adversarial_reasoning.runner --config configs/smoke.yaml
```

## Reproducibility

All runs record: torch/transformers/torchattacks/ART versions, Ollama model digests, fixed seeds. See `docs/reproducibility.md`.

## Ethics

**This benchmark is a defensive research tool.** Do not deploy in clinical settings. All attacks are sandboxed — no real-world clinical system is targeted. See `docs/ethics.md`.

## License

Code: MIT. Data sources carry their own licenses (ProstateX via TCIA DUA, VQA-RAD CC-BY, SLAKE CC-BY-SA). See `LICENSE`.

## Citation

See `CITATION.cff`.
