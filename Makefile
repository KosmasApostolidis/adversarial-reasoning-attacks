.PHONY: help install install-dev test test-fast test-gpu lint format figures \
        smoke-runs lock lock-check clean

PY    ?= python
PIP   ?= $(PY) -m pip
PYTEST ?= $(PY) -m pytest

help:
	@echo "Targets:"
	@echo "  install         pip install -e ."
	@echo "  install-dev     pip install -e .[dev] + pre-commit hooks"
	@echo "  test            full pytest suite (excludes gpu/slow)"
	@echo "  test-fast       only -m smoke (quick gate)"
	@echo "  test-gpu        only -m gpu (local CUDA only)"
	@echo "  lint            ruff check src/ scripts/ tests/ + mypy src/"
	@echo "  format          ruff format + ruff check --fix"
	@echo "  figures         regenerate paper/figures via adreason-figures"
	@echo "  smoke-runs      run 5 smoke configs (single sample, seed=0)"
	@echo "  lock            regenerate requirements.lock from pyproject.toml"
	@echo "  lock-check      verify requirements.lock is in sync with pyproject.toml"
	@echo "  clean           remove build/, dist/, *.egg-info, __pycache__"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .[dev]
	pre-commit install

test:
	$(PYTEST) -q -m "not gpu and not slow"

test-fast:
	$(PYTEST) -q -m smoke

test-gpu:
	$(PYTEST) -q -m gpu

lint:
	ruff check src/ scripts/ tests/
	mypy src/

format:
	ruff format src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

figures:
	adreason-figures paper
	adreason-figures hero
	adreason-figures attack-landscape

smoke-runs:
	@for cfg in pgd_smoke apgd_smoke targeted_tool_smoke trajectory_drift_smoke apgd_smoke_llava; do \
	  echo "=== $$cfg ==="; \
	  adreason --config configs/$$cfg.yaml --split val --mode $${cfg%_smoke*} || exit 1; \
	done

lock:
	uv pip compile pyproject.toml --extra dev --output-file requirements.lock

lock-check:
	@uv pip compile pyproject.toml --extra dev --output-file requirements.lock.check >/dev/null 2>&1
	@grep -v '^#' requirements.lock > .lock.expected
	@grep -v '^#' requirements.lock.check > .lock.actual
	@if ! diff -q .lock.expected .lock.actual >/dev/null; then \
	  echo "ERROR: requirements.lock is out of sync with pyproject.toml. Run 'make lock'."; \
	  diff -u .lock.expected .lock.actual | head -40; \
	  rm -f requirements.lock.check .lock.expected .lock.actual; \
	  exit 1; \
	fi
	@rm -f requirements.lock.check .lock.expected .lock.actual
	@echo "requirements.lock is in sync."

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache -prune -exec rm -rf {} +
	find . -type d -name .mypy_cache -prune -exec rm -rf {} +
