"""Compare PGD vs uniform-noise edit-distance for matched ε on Qwen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", required=True, help="records.jsonl from noise-mode run")
    ap.add_argument("--pgd", required=True, help="records.jsonl from PGD-mode run")
    ap.add_argument("--out", default="paper/figures/pgd_vs_noise")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    noise = _load(Path(args.noise))
    pgd = _load(Path(args.pgd))
    nd = np.array([r["edit_distance_norm"] for r in noise])
    pd_ = np.array([r["edit_distance_norm"] for r in pgd])
    eps = pgd[0]["epsilon"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(
        [nd, pd_],
        labels=["uniform noise", "PGD-L∞ (20 steps)"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#888888")
    bp["boxes"][1].set_facecolor("#c62828")
    ax.set_ylabel("normalized trajectory edit distance", fontsize=11)
    ax.set_title(
        f"PGD vs noise on Qwen2.5-VL-7B at ε={eps:.4f} (n={len(pd_)})",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for i, (label, arr) in enumerate([("noise", nd), ("PGD", pd_)], start=1):
        ax.text(i, max(arr.max(), 0.05) + 0.04,
                f"mean={arr.mean():.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "pgd_vs_noise_box.png", dpi=140)
    plt.close(fig)

    print(f"[compare_pgd_noise] noise mean={nd.mean():.3f}  PGD mean={pd_.mean():.3f}  Δ={pd_.mean()-nd.mean():+.3f}")
    print(f"[compare_pgd_noise] wrote → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
