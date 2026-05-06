"""Figure 3: tool-substitution matrix under PGD."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._common import C_BENIGN, C_PGD, OUT, _panel_label, despine, load_records


def fig3_tool_heatmap() -> None:
    pgd_recs = load_records("runs/main/pgd/records.jsonl")

    # Collect min-edit alignment: for each sample, align benign → attacked
    # and count (benign_tool, attacked_tool) substitution pairs.
    all_tools: set[str] = set()
    for r in pgd_recs:
        all_tools.update(r["benign"]["tool_sequence"])
        all_tools.update(r["attacked"]["tool_sequence"])
    tools = sorted(all_tools)
    idx = {t: i for i, t in enumerate(tools)}
    n = len(tools)

    counts = np.zeros((n, n), dtype=int)
    insert = np.zeros(n, dtype=int)  # in attacked but not in benign position
    delete = np.zeros(n, dtype=int)  # in benign but not in attacked

    for r in pgd_recs:
        b = r["benign"]["tool_sequence"]
        a = r["attacked"]["tool_sequence"]
        # Simple prefix alignment (visual approximation)
        for bi, tool in enumerate(b):
            if bi < len(a):
                if b[bi] != a[bi]:
                    counts[idx[b[bi]], idx[a[bi]]] += 1
            else:
                delete[idx[tool]] += 1
        for ai in range(len(b), len(a)):
            insert[idx[a[ai]]] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]})
    fig.subplots_adjust(wspace=0.4)

    # Heatmap
    ax = axes[0]
    vmax = max(counts.max(), 1)
    im = ax.imshow(counts, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels([t.replace("_", "\n") for t in tools], fontsize=7.5, rotation=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([t.replace("_", "\n") for t in tools], fontsize=7.5)
    ax.set_xlabel("Attacked trajectory tool")
    ax.set_ylabel("Benign trajectory tool")
    ax.set_title("Tool substitution matrix under PGD-L∞", pad=8)
    for i in range(n):
        for j in range(n):
            v = counts[i, j]
            if v > 0:
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if v > vmax * 0.5 else "black",
                )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("substitution count", fontsize=9)

    # Insert / delete bar chart
    ax2 = axes[1]
    y = np.arange(n)
    ax2.barh(y - 0.2, delete, 0.38, color=C_BENIGN, label="deleted (benign→∅)", alpha=0.8)
    ax2.barh(y + 0.2, insert, 0.38, color=C_PGD, label="inserted (∅→attacked)", alpha=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([t.replace("_", "\n") for t in tools], fontsize=7.5)
    ax2.set_xlabel("Count")
    ax2.set_title("Insertions & deletions", pad=8)
    ax2.legend(loc="lower right")
    despine(ax2)

    _panel_label(axes[0], "A")
    _panel_label(axes[1], "B")

    fig.suptitle(
        "PGD-induced tool-call perturbation anatomy (n=5 patients)", fontsize=12, fontweight="bold"
    )
    fig.savefig(OUT / "fig3_tool_heatmap.png")
    plt.close(fig)
    print("fig3 ✓")
