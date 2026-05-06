"""Graph 3: polar chart — each patient's tool sequence as radial arcs."""

from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ._common import C_NODE, OUT, _load, _short


def graph3_radial_trajectories() -> None:
    pgd_recs = _load("runs/pgd_smoke/records.jsonl")
    noise_recs = {r["sample_id"]: r for r in _load("runs/smoke/records.jsonl")}

    all_tools: list[str] = sorted(
        {
            t
            for r in pgd_recs
            for seq in [r["benign"]["tool_sequence"], r["attacked"]["tool_sequence"]]
            for t in seq
        }
    )
    n_tools = len(all_tools)
    tool_angle = {t: 2 * np.pi * i / n_tools for i, t in enumerate(all_tools)}

    n_patients = len(pgd_recs)
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("#0d1117")

    axes = []
    for i in range(n_patients):
        ax = fig.add_subplot(1, n_patients, i + 1, polar=True)
        ax.set_facecolor("#0d1117")
        axes.append(ax)

    CMAP_B = cm.Blues
    CMAP_N = cm.Greens
    CMAP_P = cm.Reds

    for ax, rec in zip(axes, pgd_recs, strict=False):
        sid = rec["sample_id"]
        nr = noise_recs.get(sid)
        seqs = {
            "benign": rec["benign"]["tool_sequence"],
            "noise": nr["attacked"]["tool_sequence"] if nr else [],
            "PGD": rec["attacked"]["tool_sequence"],
        }
        cmaps = {"benign": CMAP_B, "noise": CMAP_N, "PGD": CMAP_P}
        radii = {"benign": 0.72, "noise": 0.86, "PGD": 1.00}

        for condition, seq in seqs.items():
            if not seq:
                continue
            cmap = cmaps[condition]
            r0 = radii[condition]
            n = len(seq)
            for step_i, tool in enumerate(seq):
                ang = tool_angle[tool]
                frac = step_i / max(n - 1, 1)
                color = cmap(0.4 + 0.55 * frac)
                ax.plot(ang, r0, "o", markersize=9 * r0, color=color, alpha=0.9, zorder=4)
                if step_i > 0:
                    prev_ang = tool_angle[seq[step_i - 1]]
                    ax.annotate(
                        "",
                        xy=(ang, r0),
                        xycoords="data",
                        xytext=(prev_ang, r0),
                        textcoords="data",
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=color,
                            lw=1.4,
                            connectionstyle="arc3,rad=0.3",
                        ),
                        zorder=3,
                    )

        # Angular ticks = tool names
        ax.set_thetagrids(
            [np.degrees(tool_angle[t]) for t in all_tools],
            labels=[_short(t) for t in all_tools],
            fontsize=7,
            color="#8b949e",
        )
        ax.set_ylim(0, 1.15)
        ax.set_yticks([])
        ax.grid(color="#30363d", linewidth=0.5)
        pid = sid.split("_p")[1] if "_p" in sid else sid
        pgd_ed = rec["edit_distance_norm"]
        noise_ed = nr["edit_distance_norm"] if nr else float("nan")
        ax.set_title(
            f"P{pid}\nnoise={noise_ed:.2f}  PGD={pgd_ed:.2f}",
            fontsize=8.5,
            color=C_NODE,
            pad=14,
        )

    # Legend
    handles = [
        mpatches.Patch(color=CMAP_B(0.7), label="Benign"),
        mpatches.Patch(color=CMAP_N(0.7), label="Noise"),
        mpatches.Patch(color=CMAP_P(0.7), label="PGD"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        "Radial tool-trajectory chart: tool angles × trajectory rings",
        fontsize=13,
        fontweight="bold",
        color=C_NODE,
        y=1.01,
    )
    fig.savefig(OUT / "graph3_radial_trajectories.png")
    plt.close(fig)
    print("graph3 ✓")
