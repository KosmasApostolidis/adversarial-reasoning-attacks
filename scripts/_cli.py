"""Shared CLI helpers for figure-generation scripts.

Defaults match existing hard-coded paths so calling a migrated script with
no args reproduces today's behaviour.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_OUT_DIR = Path("paper/figures")
DEFAULT_DPI = 200
DEFAULT_FORMAT = "png"


def base_parser(
    description: str | None = None,
    *,
    runs_dir_default: Path = DEFAULT_RUNS_DIR,
    out_dir_default: Path = DEFAULT_OUT_DIR,
    dpi_default: int = DEFAULT_DPI,
    format_default: str = DEFAULT_FORMAT,
) -> argparse.ArgumentParser:
    """Build the shared argparse skeleton.

    Scripts add their own per-figure args via ``parser.add_argument(...)``
    on the returned object before ``parse_args``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=runs_dir_default,
        help="Root directory containing runs/<name>/records.jsonl trees.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=out_dir_default,
        help="Root directory for figure outputs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=dpi_default,
        help="Output DPI for raster figures.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=format_default,
        help="Output image format (png, pdf, svg, ...).",
    )
    return parser
