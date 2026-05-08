# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
