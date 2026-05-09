"""Tests for the lazy NLI judge."""

from __future__ import annotations

import pytest

from adversarial_reasoning.metrics import nli


@pytest.fixture(autouse=True)
def _reset_nli():
    nli.reset_nli()
    yield
    nli.reset_nli()


def test_set_nli_overrides_real_loader() -> None:
    def stub(p: str, h: str) -> float:
        return 0.42

    nli.set_nli(stub)
    fn = nli.get_nli()
    assert fn("premise", "hypothesis") == 0.42


def test_get_nli_returns_callable() -> None:
    nli.set_nli(lambda p, h: 0.5)
    fn = nli.get_nli()
    assert callable(fn)


def test_set_nli_none_then_get_falls_back_to_real_loader(monkeypatch) -> None:
    sentinel = object()

    def fake_build():
        def f(p: str, h: str) -> float:
            return 0.7

        return f

    monkeypatch.setattr(nli, "_build_real_nli", fake_build)
    nli.set_nli(None)
    fn = nli.get_nli()
    assert fn("a", "b") == 0.7


def test_reset_clears_override() -> None:
    nli.set_nli(lambda p, h: 1.0)
    nli.reset_nli()
    assert nli._OVERRIDE is None


def test_nli_model_id_constant() -> None:
    assert nli.NLI_MODEL_ID == "cross-encoder/nli-deberta-v3-large"
