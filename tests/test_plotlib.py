"""Smoke tests for shared figure-script plotlib helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _plotlib import despine, load_records, panel_label, tool_palette  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def tmp_records(tmp_path: Path) -> Path:
    p = tmp_path / "rec.jsonl"
    _write_jsonl(p, [{"id": "a", "x": 1}, {"id": "b", "x": 2}])
    return p


def test_load_records_basic(tmp_records: Path) -> None:
    out = load_records(tmp_records)
    assert out == [{"id": "a", "x": 1}, {"id": "b", "x": 2}]


def test_load_records_skips_missing_with_warning(
    tmp_path: Path, tmp_records: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "nope.jsonl"
    with caplog.at_level("WARNING", logger="_plotlib"):
        out = load_records(missing, tmp_records)
    assert len(out) == 2, "missing file must still be skipped, not crash"
    assert any("missing file" in r.message and "nope.jsonl" in r.message for r in caplog.records), (
        "skipping a missing file must emit a WARNING log line"
    )


def test_load_records_strict_raises_when_empty(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(ValueError, match="0 records"):
        load_records(missing, strict=True)


def test_load_records_strict_passes_when_records_present(
    tmp_records: Path,
) -> None:
    out = load_records(tmp_records, strict=True)
    assert len(out) == 2


def test_load_records_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "blank.jsonl"
    p.write_text('{"id": "a"}\n\n   \n{"id": "b"}\n', encoding="utf-8")
    out = load_records(p)
    assert [r["id"] for r in out] == ["a", "b"]


def test_load_records_concat_multi(tmp_path: Path) -> None:
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_jsonl(p1, [{"id": "a"}])
    _write_jsonl(p2, [{"id": "b"}, {"id": "c"}])
    out = load_records(p1, p2)
    assert [r["id"] for r in out] == ["a", "b", "c"]


def test_despine_hides_top_and_right() -> None:
    fig, ax = plt.subplots()
    try:
        despine(ax)
        assert ax.spines["top"].get_visible() is False
        assert ax.spines["right"].get_visible() is False
        assert ax.spines["left"].get_visible() is True
    finally:
        plt.close(fig)


def test_despine_selective() -> None:
    fig, ax = plt.subplots()
    try:
        despine(ax, top=False, right=True)
        assert ax.spines["top"].get_visible() is True
        assert ax.spines["right"].get_visible() is False
    finally:
        plt.close(fig)


def test_panel_label_adds_text() -> None:
    fig, ax = plt.subplots()
    try:
        n_before = len(ax.texts)
        panel_label(ax, "A")
        assert len(ax.texts) == n_before + 1
        assert ax.texts[-1].get_text() == "A"
    finally:
        plt.close(fig)


def test_tool_palette_stable_unsorted() -> None:
    pal = tool_palette(["search_pubmed", "lookup_guideline", "abc"])
    assert list(pal) == ["search_pubmed", "lookup_guideline", "abc"]
    assert all(len(c) == 4 for c in pal.values())  # rgba tuples


def test_tool_palette_sorted() -> None:
    pal = tool_palette(["c", "a", "b"], sort=True)
    assert list(pal) == ["a", "b", "c"]


def test_tool_palette_distinct_colors() -> None:
    pal = tool_palette([f"t{i}" for i in range(5)])
    colors = list(pal.values())
    # tab20 first 5 entries are pairwise distinct
    assert len(set(colors)) == len(colors)
