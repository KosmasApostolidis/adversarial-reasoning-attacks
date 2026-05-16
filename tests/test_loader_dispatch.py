"""Tests for models.loader family dispatch + Ollama lookup.

Patch the lazy-imported wrapper classes so we exercise the dispatch
without loading multi-GB HF weights.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from adversarial_reasoning.models import loader as loader_mod


@pytest.fixture(autouse=True)
def _allow_mutable_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    # Dispatch tests don't pin SHAs — bypass the supply-chain guard.
    monkeypatch.setenv("ADREASON_ALLOW_MUTABLE_HF_REVISION", "1")


def _write_models_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(
        textwrap.dedent(
            """
            models:
              qwen_x:
                family: qwen_vl
                hf_id: dummy/qwen
              llava_x:
                family: llava_next
                hf_id: dummy/llava
              internvl_x:
                family: internvl2
                hf_id: dummy/internvl
              dead_llama:
                family: llama_vision
                hf_id: dummy/llama
              alien_x:
                family: martian_vlm
                hf_id: dummy/martian
              ollama_only:
                family: qwen_vl
                hf_id: dummy/q
                ollama_tag: q:tag
              no_ollama:
                family: qwen_vl
                hf_id: dummy/q
            """
        ).strip()
    )
    return p


def _patch_family(monkeypatch: pytest.MonkeyPatch, modname: str, classname: str) -> MagicMock:
    """Inject a fake submodule with a constructor we can assert on."""
    cls = MagicMock(name=classname, return_value=MagicMock(name=f"{classname}-instance"))
    fake_mod = ModuleType(f"adversarial_reasoning.models.{modname}")
    setattr(fake_mod, classname, cls)
    monkeypatch.setitem(sys.modules, f"adversarial_reasoning.models.{modname}", fake_mod)
    return cls


def test_dispatch_qwen_vl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_models_yaml(tmp_path)
    cls = _patch_family(monkeypatch, "qwen_vl", "QwenVL")
    loader_mod.load_hf_vlm("qwen_x", config_path=cfg)
    cls.assert_called_once()
    kwargs = cls.call_args.kwargs
    assert kwargs["hf_id"] == "dummy/qwen"


def test_dispatch_llava(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_models_yaml(tmp_path)
    cls = _patch_family(monkeypatch, "llava", "LlavaNext")
    loader_mod.load_hf_vlm("llava_x", config_path=cfg)
    assert cls.called


def test_dispatch_internvl2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_models_yaml(tmp_path)
    cls = _patch_family(monkeypatch, "internvl2", "InternVL2")
    loader_mod.load_hf_vlm("internvl_x", config_path=cfg)
    assert cls.called


def test_removed_llama_vision_family_raises(tmp_path: Path) -> None:
    """Regression guard: llama_vision was removed from the eval surface."""
    cfg = _write_models_yaml(tmp_path)
    with pytest.raises(ValueError, match="Unknown VLM family: llama_vision"):
        loader_mod.load_hf_vlm("dead_llama", config_path=cfg)


def test_unknown_family_raises(tmp_path: Path) -> None:
    cfg = _write_models_yaml(tmp_path)
    with pytest.raises(ValueError, match="Unknown VLM family"):
        loader_mod.load_hf_vlm("alien_x", config_path=cfg)


def test_load_ollama_vlm_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _write_models_yaml(tmp_path)
    fake_client_cls = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(loader_mod, "OllamaVLMClient", fake_client_cls)
    loader_mod.load_ollama_vlm("ollama_only", config_path=cfg)
    fake_client_cls.assert_called_once()
    assert fake_client_cls.call_args.kwargs["ollama_tag"] == "q:tag"
    assert fake_client_cls.call_args.kwargs["family"] == "qwen_vl"


def test_load_ollama_vlm_missing_tag_raises(tmp_path: Path) -> None:
    cfg = _write_models_yaml(tmp_path)
    with pytest.raises(NotImplementedError, match="No Ollama image"):
        loader_mod.load_ollama_vlm("no_ollama", config_path=cfg)
