"""Cross-model comparison figures (Qwen vs LLaVA at matched ε)."""

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
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--llava", required=True)
    ap.add_argument("--out", default="paper/figures/cross_model")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    qwen = _load(Path(args.qwen))
    llava = _load(Path(args.llava))

    qd = np.array([r["edit_distance_norm"] for r in qwen])
    ld = np.array([r["edit_distance_norm"] for r in llava])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot([qd, ld], labels=["Qwen2.5-VL-7B", "LLaVA-v1.6-Mistral-7B"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#4a7ab8")
    bp["boxes"][1].set_facecolor("#c89a3a")
    ax.set_ylabel("normalized trajectory edit distance", fontsize=11)
    ax.set_title(
        f"Cross-model attack sensitivity at ε={qwen[0]['epsilon']:.4f} "
        f"(uniform-noise mode, n={len(qd)} per model)",
        fontsize=11,
    )
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0,
               label="noise floor (T=0 deterministic)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9)
    for i, (label, arr) in enumerate(
        [("Qwen", qd), ("LLaVA", ld)], start=1
    ):
        ax.text(i, arr.max() + 0.04, f"mean={arr.mean():.3f}",
                ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "cross_model_edit_distance.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    x = np.arange(len(qwen))
    ax.bar(x - width / 2, qd, width, label="Qwen2.5-VL", color="#4a7ab8",
           edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, ld, width, label="LLaVA-v1.6", color="#c89a3a",
           edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r["sample_id"].split("_p")[1] for r in qwen],
                       rotation=0, fontsize=9)
    ax.set_xlabel("patient id", fontsize=11)
    ax.set_ylabel("normalized edit distance", fontsize=11)
    ax.set_title(f"Per-patient attack sensitivity at ε={qwen[0]['epsilon']:.4f}",
                 fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "cross_model_per_patient.png", dpi=140)
    plt.close(fig)

    print(f"[compare_figures] wrote → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
