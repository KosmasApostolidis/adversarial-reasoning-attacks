"""Coverage tests for runner.cli — argparse, --dry-run, error paths.

These tests exercise the CLI without loading any model. The --dry-run flag
short-circuits before model load, which is the wedge that makes the path
unit-testable without GPU / heavy deps.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from adversarial_reasoning.runner import cli


def _write_minimal_config(tmp_path: Path) -> Path:
    p = tmp_path / "exp.yaml"
    p.write_text(
        textwrap.dedent(
            """
            experiment:
              name: t
              models: [m]
              tasks: [t]
              attacks: [pgd_linf]
              seeds: [0]
              epsilons_linf: [0.01]
            """
        ).strip()
    )
    return p


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ei:
        cli.main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "Adversarial reasoning runner" in out
    assert "--dry-run" in out


def test_dry_run_prints_repr_and_exits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _write_minimal_config(tmp_path)
    rc = cli.main(["--config", str(cfg), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RunnerConfig(" in out
    assert "name='t'" in out


def test_dry_run_with_extends(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("experiment: {seeds: [42]}\n")
    child = tmp_path / "child.yaml"
    child.write_text(
        textwrap.dedent(
            """
            _extends: base.yaml
            experiment:
              name: child
              models: [m]
              tasks: [t]
              attacks: [pgd_linf]
              epsilons_linf: [0.01]
            """
        ).strip()
    )
    rc = cli.main(["--config", str(child), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "name='child'" in out
    assert "seeds=[42]" in out


def test_missing_config_arg_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ei:
        cli.main([])
    assert ei.value.code != 0
    err = capsys.readouterr().err
    assert "--config" in err


def test_bad_mode_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _write_minimal_config(tmp_path)
    with pytest.raises(SystemExit) as ei:
        cli.main(["--config", str(cfg), "--mode", "not_a_mode"])
    assert ei.value.code != 0
    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_bad_config_path_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        cli.main(["--config", str(bogus), "--dry-run"])


def test_existing_records_without_overwrite_returns_one(tmp_path: Path) -> None:
    cfg = _write_minimal_config(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "records.jsonl").write_text("{}\n")
    rc = cli.main(["--config", str(cfg), "--out", str(out_dir)])
    assert rc == 1


def _stubbed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dummy_image,
    *,
    extra_args: list[str] | None = None,
):
    """Run cli.main on the noise path with all heavy deps stubbed."""
    from adversarial_reasoning.agents.base import ToolCall, Trajectory
    from adversarial_reasoning.runner import cli as cli_mod
    from adversarial_reasoning.tasks.loader import TaskSample

    sample = TaskSample(task_id="t", sample_id="s0", image=dummy_image, prompt="?")

    class StubAgent:
        def __init__(self, *_, **__) -> None:
            pass

        def run(self, *, task_id, image, prompt, seed, max_steps):
            return Trajectory(
                task_id=task_id,
                model_id="stub",
                seed=seed,
                tool_calls=[ToolCall(step=0, name="query_guidelines", args={})],
                final_answer="",
            )

    monkeypatch.setattr(cli_mod, "load_hf_vlm", lambda key: object())
    monkeypatch.setattr(cli_mod, "MedicalAgent", StubAgent)
    monkeypatch.setattr(cli_mod, "default_registry", lambda: {})
    monkeypatch.setattr(cli_mod, "load_task", lambda *a, **kw: iter([sample]))
    # Bypass attacks.yaml — pass a stubbed loader that returns nothing useful.
    # We're on the noise path so resolve_epsilons reads from cfg.epsilons_linf.
    monkeypatch.setattr(cli_mod, "_load_yaml", lambda _: {"attacks": {"pgd_linf": {}}})
    cfg = _write_minimal_config(tmp_path)
    out_dir = tmp_path / "out"
    args = ["--config", str(cfg), "--out", str(out_dir), "--mode", "noise"]
    if extra_args:
        args.extend(extra_args)
    rc = cli_mod.main(args)
    return rc, out_dir


def test_noise_loop_writes_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dummy_image
) -> None:
    rc, out_dir = _stubbed_run(tmp_path, monkeypatch, dummy_image)
    assert rc == 0
    records = (out_dir / "records.jsonl").read_text().strip().splitlines()
    assert len(records) == 1
    summary = (out_dir / "summary.json").read_text()
    assert '"records": 1' in summary
    assert '"errors": 0' in summary


def test_loop_records_error_and_returns_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dummy_image
) -> None:
    """Agent.run raising → counted as error, return code 2, summary records it."""
    from adversarial_reasoning.runner import cli as cli_mod
    from adversarial_reasoning.tasks.loader import TaskSample

    sample = TaskSample(task_id="t", sample_id="s0", image=dummy_image, prompt="?")

    class FailingAgent:
        def __init__(self, *_, **__) -> None:
            pass

        def run(self, **_):
            raise RuntimeError("boom")

    monkeypatch.setattr(cli_mod, "load_hf_vlm", lambda key: object())
    monkeypatch.setattr(cli_mod, "MedicalAgent", FailingAgent)
    monkeypatch.setattr(cli_mod, "default_registry", lambda: {})
    monkeypatch.setattr(cli_mod, "load_task", lambda *a, **kw: iter([sample]))
    monkeypatch.setattr(cli_mod, "_load_yaml", lambda _: {"attacks": {"pgd_linf": {}}})
    cfg = _write_minimal_config(tmp_path)
    out_dir = tmp_path / "out"
    rc = cli_mod.main(["--config", str(cfg), "--out", str(out_dir), "--mode", "noise"])
    assert rc == 2
    summary = (out_dir / "summary.json").read_text()
    assert '"errors": 1' in summary
    assert '"records": 0' in summary


def test_loop_skips_when_no_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from adversarial_reasoning.runner import cli as cli_mod

    class StubAgent:
        def __init__(self, *_, **__) -> None:
            pass

    monkeypatch.setattr(cli_mod, "load_hf_vlm", lambda key: object())
    monkeypatch.setattr(cli_mod, "MedicalAgent", StubAgent)
    monkeypatch.setattr(cli_mod, "default_registry", lambda: {})
    monkeypatch.setattr(cli_mod, "load_task", lambda *a, **kw: iter([]))
    monkeypatch.setattr(cli_mod, "_load_yaml", lambda _: {"attacks": {"pgd_linf": {}}})
    cfg = _write_minimal_config(tmp_path)
    rc = cli_mod.main(["--config", str(cfg), "--out", str(tmp_path / "out"), "--mode", "noise"])
    assert rc == 0
    assert "no samples" in capsys.readouterr().out


def test_overwrite_allows_existing_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dummy_image
) -> None:
    rc, _ = _stubbed_run(tmp_path, monkeypatch, dummy_image)
    assert rc == 0
    # Second run with --overwrite must succeed despite preexisting records.jsonl
    rc2, _ = _stubbed_run(tmp_path, monkeypatch, dummy_image, extra_args=["--overwrite"])
    assert rc2 == 0
