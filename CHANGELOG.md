# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
