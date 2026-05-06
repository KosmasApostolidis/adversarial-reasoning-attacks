"""Extra coverage for gates.preprocessing_transfer: write_gate_report + _cli."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from adversarial_reasoning.gates import preprocessing_transfer as ptm
from adversarial_reasoning.gates.preprocessing_transfer import (
    PreprocessingTransferResult,
    _cli,
    write_gate_report,
)


def test_write_gate_report_creates_file_and_writes_fields(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "report.txt"
    result = PreprocessingTransferResult(
        model_name="m",
        epsilon_requested=0.05,
        effective_linf_post_roundtrip=0.04,
        gate_threshold=0.01,
        passed=True,
    )
    write_gate_report(result, out)
    text = out.read_text(encoding="utf-8")
    assert "model: m" in text
    assert "passed: True" in text
    assert "gate_threshold: 0.010000" in text


def test_cli_with_synthetic_image_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "gate.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocessing_transfer",
            "--model-name",
            "stub-model",
            "--epsilon",
            "0.0625",
            "--gate-threshold",
            "0.001",
            "--out",
            str(out),
        ],
    )
    rc = _cli()
    assert rc == 0
    captured = capsys.readouterr().out
    assert "[gate:preprocessing_transfer]" in captured
    assert out.exists()


def test_cli_with_explicit_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image as _Img

    img_path = tmp_path / "img.png"
    _Img.new("RGB", (32, 32), color=(10, 20, 30)).save(img_path)
    out = tmp_path / "gate.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocessing_transfer",
            "--image",
            str(img_path),
            "--out",
            str(out),
        ],
    )
    rc = _cli()
    assert rc in (0, 1)
    assert out.exists()


def test_module_keeps_torch_reference_alive() -> None:
    # The module imports torch defensively; sanity-check the reference exists.
    assert ptm._ is not None
