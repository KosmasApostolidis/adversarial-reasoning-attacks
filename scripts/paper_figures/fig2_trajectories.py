"""Figure 2: Gantt-style benign / noise / PGD sequences (top-3 drifted patients)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ._common import C_NOISE, C_PGD, OUT, _tool_palette, load_records


def fig2_trajectories() -> None:
    noise_recs = load_records("runs/main/noise/records.jsonl")
    pgd_recs = load_records("runs/main/pgd/records.jsonl")

    # Pick 3 most interesting samples (highest PGD edit distance)
    by_ed = sorted(pgd_recs, key=lambda r: r["edit_distance_norm"], reverse=True)[:3]

    # Build tool palette from all seqs
    all_tools: set[str] = set()
    noise_map = {r["sample_id"]: r for r in noise_recs}
    for r in by_ed:
        all_tools.update(r["benign"]["tool_sequence"])
        all_tools.update(r["attacked"]["tool_sequence"])
        nr = noise_map.get(r["sample_id"])
        if nr:
            all_tools.update(nr["attacked"]["tool_sequence"])
    pal = _tool_palette(list(all_tools))

    n_samples = len(by_ed)
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3.2 * n_samples))
    if n_samples == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.55)

    ROW_LABELS = ["Benign", "Noise", "PGD"]
    ROW_COLORS = ["#1a9850", C_NOISE, C_PGD]

    for ax, rec in zip(axes, by_ed, strict=False):
        sid = rec["sample_id"]
        nr = noise_map.get(sid)
        seqs = [
            rec["benign"]["tool_sequence"],
            nr["attacked"]["tool_sequence"] if nr else [],
            rec["attacked"]["tool_sequence"],
        ]
        max_len = max(len(s) for s in seqs)

        for row_i, (label, seq, rc) in enumerate(zip(ROW_LABELS, seqs, ROW_COLORS, strict=False)):
            for col, tool in enumerate(seq):
                color = pal[tool]
                ax.add_patch(
                    FancyBboxPatch(
                        (col + 0.04, row_i - 0.38),
                        0.84,
                        0.76,
                        boxstyle="round,pad=0.02",
                        facecolor=color,
                        edgecolor="white",
                        linewidth=1.2,
                        zorder=2,
                    )
                )
                short = tool.replace("_", "\n")
                ax.text(
                    col + 0.46,
                    row_i,
                    short,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    fontweight="bold",
                    zorder=3,
                )
            ax.text(
                -0.4,
                row_i,
                label,
                ha="right",
                va="center",
                fontsize=9,
                color=rc,
                fontweight="bold",
            )

        ax.set_xlim(-0.6, max_len + 0.2)
        ax.set_ylim(-0.7, len(seqs) - 0.3)
        ax.set_yticks([])
        ax.set_xticks(range(max_len))
        ax.set_xticklabels([f"step {i + 1}" for i in range(max_len)], fontsize=8)
        pid = sid.split("_p")[1] if "_p" in sid else sid
        pgd_ed = rec["edit_distance_norm"]
        n_ed = nr["edit_distance_norm"] if nr else float("nan")
        ax.set_title(
            f"Patient {pid}   |   edit dist: noise={n_ed:.3f}  PGD={pgd_ed:.3f}",
            fontsize=10,
            loc="left",
            pad=5,
        )
        ax.spines[:].set_visible(False)
        ax.axhline(-0.5, color="#dddddd", linewidth=0.8)

    # Shared legend
    handles = [
        mpl.patches.Patch(facecolor=pal[t], label=t.replace("_", " ")) for t in sorted(all_tools)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(all_tools), 4),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        title="Tools",
        title_fontsize=9,
    )
    fig.suptitle(
        "Tool-call trajectory comparison: benign vs noise vs PGD (top-3 drifted patients)",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(OUT / "fig2_trajectories.png")
    plt.close(fig)
    print("fig2 ✓")
