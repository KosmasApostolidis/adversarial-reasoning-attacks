"""Mock-based tests for OllamaVLMClient + OllamaSettings.

We never hit a live Ollama daemon — the client constructor and chat
method are exercised against a stubbed ``ollama.Client``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from adversarial_reasoning.models import ollama_client as oc_mod
from adversarial_reasoning.models.ollama_client import (
    OllamaSettings,
    OllamaVLMClient,
)


def test_settings_host_default_factory_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OllamaSettings.host must use default_factory so OLLAMA_HOST is read at
    instantiation time, not at import time (regression: see memory note)."""
    monkeypatch.setenv("OLLAMA_HOST", "http://example.com:9000")
    monkeypatch.setenv("OLLAMA_ALLOW_REMOTE", "1")
    s = OllamaSettings()
    assert s.host == "http://example.com:9000"


def test_settings_host_falls_back_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    s = OllamaSettings()
    assert s.host == "http://127.0.0.1:11434"


def test_settings_explicit_host_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://x:1")
    s = OllamaSettings(host="http://override:1")
    assert s.host == "http://override:1"


def _patch_ollama(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake_client = MagicMock(name="ollama.Client.instance")
    fake_client.chat.return_value = {
        "message": {"content": "ok"},
        "done_reason": "stop",
    }
    fake_ollama = MagicMock(name="ollama-mod")
    fake_ollama.Client.return_value = fake_client
    monkeypatch.setattr(oc_mod, "ollama", fake_ollama)
    return fake_client


def test_client_construction_uses_settings_host_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ollama(monkeypatch)
    settings = OllamaSettings(host="http://custom:11", request_timeout_s=42.0)
    OllamaVLMClient(ollama_tag="m:tag", family="qwen_vl", settings=settings)
    oc_mod.ollama.Client.assert_called_once_with(host="http://custom:11", timeout=42.0)


def test_client_raises_when_ollama_pkg_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oc_mod, "ollama", None)
    with pytest.raises(RuntimeError, match="`ollama` package not installed"):
        OllamaVLMClient(ollama_tag="m:tag", family="qwen_vl")


def test_generate_returns_text_and_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _patch_ollama(monkeypatch)
    fake_client.chat.return_value = {
        "message": {"content": "hello"},
        "done_reason": "length",
    }
    client = OllamaVLMClient(ollama_tag="m:tag", family="qwen_vl")
    img = Image.new("RGB", (8, 8), color=(0, 0, 0))
    out = client.generate(img, prompt="hi")
    assert out.text == "hello"
    assert out.finish_reason == "length"


def test_generate_default_finish_reason_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _patch_ollama(monkeypatch)
    fake_client.chat.return_value = {"message": {"content": "x"}}
    client = OllamaVLMClient(ollama_tag="m:tag", family="qwen_vl")
    img = Image.new("RGB", (8, 8), color=(255, 255, 255))
    out = client.generate(img, prompt="hi")
    assert out.finish_reason == "stop"


def test_generate_supports_gradients_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ollama(monkeypatch)
    client = OllamaVLMClient(ollama_tag="m:tag", family="qwen_vl")
    assert client.supports_gradients is False


def test_generate_threads_seed_and_max_new_tokens_into_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _patch_ollama(monkeypatch)
    client = OllamaVLMClient(ollama_tag="m:tag", family="qwen_vl")
    img = Image.new("RGB", (4, 4), color=(0, 0, 0))
    client.generate(img, prompt="hi", max_new_tokens=128, temperature=0.5, seed=42)
    kwargs = fake_client.chat.call_args.kwargs
    assert kwargs["options"] == {"temperature": 0.5, "num_predict": 128, "seed": 42}
    assert "tools" not in kwargs


def test_generate_omits_seed_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _patch_ollama(monkeypatch)
    client = OllamaVLMClient(ollama_tag="m:tag", family="qwen_vl")
    img = Image.new("RGB", (4, 4))
    client.generate(img, prompt="hi", max_new_tokens=64)
    opts = fake_client.chat.call_args.kwargs["options"]
    assert "seed" not in opts
    assert opts["num_predict"] == 64


def test_generate_forwards_tools_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _patch_ollama(monkeypatch)
    client = OllamaVLMClient(ollama_tag="m:tag", family="qwen_vl")
    img = Image.new("RGB", (4, 4))
    schema = [{"type": "function", "function": {"name": "f"}}]
    client.generate(img, prompt="hi", tools_schema=schema)
    assert fake_client.chat.call_args.kwargs["tools"] == schema


def test_chat_honors_max_retries_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _patch_ollama(monkeypatch)
    fake_client.chat.side_effect = ConnectionError("boom")
    settings = OllamaSettings(max_retries=5)
    client = OllamaVLMClient(ollama_tag="m:tag", family="qwen_vl", settings=settings)
    img = Image.new("RGB", (4, 4))
    # Avoid 1+s exponential waits during test
    monkeypatch.setattr(oc_mod, "wait_exponential", lambda **_: lambda *a, **k: 0)
    with pytest.raises(ConnectionError):
        client.generate(img, prompt="hi")
    assert fake_client.chat.call_count == 5


def test_chat_succeeds_within_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _patch_ollama(monkeypatch)
    fake_client.chat.side_effect = [
        ConnectionError("1"),
        ConnectionError("2"),
        {"message": {"content": "ok"}, "done_reason": "stop"},
    ]
    settings = OllamaSettings(max_retries=5)
    client = OllamaVLMClient(ollama_tag="m:tag", family="qwen_vl", settings=settings)
    monkeypatch.setattr(oc_mod, "wait_exponential", lambda **_: lambda *a, **k: 0)
    img = Image.new("RGB", (4, 4))
    out = client.generate(img, prompt="hi")
    assert out.text == "ok"
    assert fake_client.chat.call_count == 3
