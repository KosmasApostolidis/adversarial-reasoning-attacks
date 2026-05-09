# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — CoT-Aware Adversarial Evaluation v1 (schema 0.4.0)
- **Records schema bump 0.3.x → 0.4.0**: `runner/records.py` now
  emits `reasoning_trace` (raw per-step CoT, includes inline tool-call
  JSON) and an optional `cot_metrics` block on each pair record:
  `cot_drift_score`, `cot_faithfulness_{benign,attacked}`,
  `cot_hallucination_{benign,attacked}`, `cot_refusal_{benign,attacked}`,
  plus raw refusal probabilities. Backward compatible — legacy
  records without these fields skip CoT figures and the CoT stats
  table cleanly.
- **`metrics/nli.py`** — eager DeBERTa-v3-large-MNLI loader.
  `NLI_MODEL_REVISION` constant (env-overridable) pins the HF
  revision SHA for paper reproducibility.
- **`metrics/cot.py`** — five pure-NLI scoring functions:
  `clean_cot`, `cot_drift_score`, `cot_faithfulness`,
  `cot_hallucination`, `cot_refusal`, plus a `score_pair` convenience
  wrapper that returns the 9-key CoT metric dict consumed by
  `pair_record(cot_metrics=...)`.
- **`scripts/backfill_cot_metrics.py`** — idempotent enrichment of
  legacy `records.jsonl` to v0.4.0 (skips rows missing
  `reasoning_trace`, skips rows already scored). Produces
  `records_cot.jsonl` for the figure / table pipeline.
- **CoT figure outputs** — additive only. New panels fire only when
  records carry CoT fields (`scripts/hero/_common.has_cot`):
  - `paper/figures/hero/{cot_overlay,heatmap_drift,heatmap_faith}.png`
  - `paper/figures/attack_landscape/fig6_cot_axis.png`
  - `paper/figures/sanity/{null_distribution,cot_confusion}.png`
- **Sanity scripts** — `scripts/cot_null_distribution.py` (5-reseed
  pairwise drift floor) and `scripts/cot_confusion_matrix.py`
  (silent-corruption quadrant: tool-flip × CoT-drift).
- **Stats table extension** — `scripts/build_stats_table.py` gains a
  `--cot-out` flag that emits a sibling 8-column LaTeX table
  (CoT-drift, Δfaithfulness, Δhallucination, refusal-rate-attacked)
  with bootstrap CI + Wilcoxon + BH at q=0.05. Existing 5-column
  `main_benchmark.tex` byte-identical when CoT fields absent.
- Tests: 7 new in `tests/test_build_stats_table.py`, 13 in
  `tests/test_cot_sanity_scripts.py`, 28 in `tests/test_cot_metrics.py`,
  6 in `tests/test_backfill_cot_metrics.py`. Full suite 379 / 379;
  coverage 96.83%.

### Notes — v0.4.0 migration
- The schema bump is additive: existing records.jsonl files load
  unchanged. Run `scripts/backfill_cot_metrics.py --in <legacy.jsonl>
  --out records_cot.jsonl` once to enrich a Phase-2 sweep without
  re-running attacks. Legacy records lacking `reasoning_trace`
  (pre-v0.4 sweeps) are skipped silently — recapture by re-running
  the affected legs with the v0.4 runner.
- Out of scope for v1: text-injection attacks, system-prompt attacks,
  cross-vector attacks, hidden-thinking VLMs, span-level alluvial
  CoT-tool alignment, API-judge calibration, and CoT scoring for
  Ollama / InternVL2 / LLaVA-13B (none currently flow through the
  gradient attack loop). Each gets its own follow-up sub-project.

### Added
- `runner.cli` `--dry-run` flag — resolves config through the loader +
  schema, prints `RunnerConfig.__repr__()`, exits before any model load.
  Additive only; existing flags unchanged. Makes the runner CLI
  unit-testable without GPU / heavy deps.
- `tests/test_runner_cli.py` — covers `--help`, `--dry-run` (with and
  without `_extends`), missing/bad config path, bad mode, existing-records
  guard, full noise-loop happy path with stubbed `load_hf_vlm` /
  `MedicalAgent` / `load_task`, error-path counting, no-samples skip, and
  `--overwrite`. `runner/cli.py` 50% → 99% per-file.
- `tests/test_runner_attacks.py` — covers `perturb()` dispatch (noise +
  gradient-mode rejection + unknown), `build_attack()` table for all four
  modes, `_build_attack_target` mode dispatch, and a stubbed
  `run_gradient_attack` happy path validating tensor reshape, attention
  concat, model-family extras flow, and per-mode metadata. `runner/attacks.py`
  58% → 99% per-file.
- `src/adversarial_reasoning/runner/schema.py` — pydantic v2 `ExperimentConfig`
  with `extra="forbid"`. Catches typo'd YAML keys at load time instead of
  silently dropping them.
- `_extends:` top-level YAML key — children may pull defaults from a base
  YAML via deep-merge (child wins on conflicts; nested dicts merge; lists
  replace wholesale). Cycle-detected.
- `configs/_base.yaml` — minimal placeholder containing only
  `seeds: [0]`, which matches the runtime default. Adopting `_extends` is a
  zero-behavior-change operation for every existing config.
- `tests/test_runner_schema.py` — every committed YAML validates through
  the schema; covers `extra="forbid"` rejection, deep-merge semantics,
  cycle detection, and the `_LEGACY_CONFIG_LOADER=1` bypass.

### Changed
- `runner.config.load_runner_config` now resolves `_extends:` and validates
  through `ExperimentConfig` before constructing `RunnerConfig`. Set
  `_LEGACY_CONFIG_LOADER=1` to bypass schema validation (removed in v0.4).

### Notes
- Per-config YAML migration to `_extends: _base.yaml` is intentionally
  deferred. Drift analysis showed adopting wider base defaults would
  change runtime behavior for 28 of 31 configs (notably `task_overrides`
  for `prostate_mri_workup`). Schema validation alone is the high-value
  win in v0.3.x; broader consolidation requires per-config audit.
- No code was removed in v0.3.x. `vulture src/ --min-confidence 80`
  returned zero hits; lower-confidence candidates were all false positives
  (pydantic v2 model fields, dataclass fields, lazy `__getattr__`,
  abstract methods on `VLMBase`, gate entry points referenced from docs
  and tests). The codebase is structurally clean post-May-6 monolith
  split and the prior bug-fix rounds.

### Mypy
- Tightened global `[tool.mypy]` settings: `disallow_untyped_defs`,
  `disallow_incomplete_defs`, `check_untyped_defs`,
  `disallow_untyped_decorators`, `no_implicit_optional`,
  `warn_unused_ignores`, `warn_redundant_casts`. The full src/ tree
  passes (44 files, 0 errors). Heavy strict flags
  (`disallow_any_generics`, `disallow_untyped_calls`) are deferred to
  v0.4.x — those require typing every torch / transformers / numpy
  call site.
- Removed a redundant `cast` in `models/ollama_client.py` flagged by
  the new `warn_redundant_casts` rule.
- Annotated the `gates/__init__.__getattr__` PEP 562 lazy-import
  shim with `-> object`.

## [0.3.0] — 2026-05-08

### Added
- `requirements.lock` — pinned dependency tree generated from `pyproject.toml` via `uv pip compile`. Reproducibility floor for local + CI installs.
- `make lock` and `make lock-check` Makefile targets.
- `lock-check` job in `.github/workflows/lint.yml` — fails CI if `requirements.lock` drifts from `pyproject.toml`.
- `CHANGELOG.md` (this file).

### Changed
- `src/adversarial_reasoning/__init__.py:__version__` synced from `0.1.0` to `0.3.0` to match `pyproject.toml`. Single source of truth on release version is no longer split across two files.

### Notes
- Surgical release-prep pass. No public API change. Public exports, CLI flags, `records.jsonl` schema, and config keys are all unchanged from 0.2.0.

## [0.2.0] — 2026-04-29

Prior history. See git log.
