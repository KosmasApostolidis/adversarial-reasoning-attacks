"""Graph 4: Sankey-style flow — what happened to each benign tool."""

from __future__ import annotations

from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ._common import C_BENIGN, C_NODE, OUT, _load, _short


def graph4_sankey() -> None:
    pgd_recs = _load("runs/pgd_smoke/records.jsonl")

    all_tools: list[str] = sorted(
        {
            t
            for r in pgd_recs
            for seq in [r["benign"]["tool_sequence"], r["attacked"]["tool_sequence"]]
            for t in seq
        }
    )
    n = len(all_tools)

    # Count how many times each benign tool appeared, and for each position
    # whether it was kept, substituted, or dropped in the attacked sequence
    kept_ct: Counter = Counter()
    sub_ct: Counter = Counter()
    drop_ct: Counter = Counter()
    ins_ct: Counter = Counter()

    for r in pgd_recs:
        b = r["benign"]["tool_sequence"]
        a = r["attacked"]["tool_sequence"]
        for i, bt in enumerate(b):
            if i < len(a):
                if bt == a[i]:
                    kept_ct[bt] += 1
                else:
                    sub_ct[bt] += 1
            else:
                drop_ct[bt] += 1
        for i in range(len(b), len(a)):
            ins_ct[a[i]] += 1

    fig, ax = plt.subplots(figsize=(13, 7))

    X_LEFT, X_RIGHT = 0.15, 0.85
    ys = np.linspace(0.1, 0.9, n)
    tool_y = {t: ys[i] for i, t in enumerate(all_tools)}
    bar_h = 0.06

    COLORS = {
        "kept": "#3fb950",
        "sub": "#f85149",
        "drop": "#8b949e",
        "ins": "#d2a8ff",
    }

    # Left column — benign tool boxes
    for t in all_tools:
        y = tool_y[t]
        total = kept_ct[t] + sub_ct[t] + drop_ct[t]
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (X_LEFT - 0.09, y - bar_h / 2),
                0.08,
                bar_h,
                boxstyle="round,pad=0.005",
                facecolor="#21262d",
                edgecolor=C_BENIGN,
                linewidth=1.5,
            )
        )
        ax.text(
            X_LEFT - 0.05,
            y,
            _short(t),
            ha="center",
            va="center",
            fontsize=7.5,
            color=C_NODE,
            fontweight="bold",
        )
        if total > 0:
            ax.text(
                X_LEFT - 0.12, y, str(total), ha="right", va="center", fontsize=8, color="#8b949e"
            )

    # Right column — fate boxes stacked
    {t: tool_y[t] for t in all_tools}
    fate_entries = []  # (left_tool, x_right, y_right, color, w)

    for lt in all_tools:
        # kept
        if kept_ct[lt]:
            fate_entries.append((lt, lt, kept_ct[lt], "kept"))
        # substituted → target tool
        if sub_ct[lt]:
            # pick first attacked tool at same position
            for r in pgd_recs:
                b = r["benign"]["tool_sequence"]
                a = r["attacked"]["tool_sequence"]
                for i, bt in enumerate(b):
                    if bt == lt and i < len(a) and a[i] != lt:
                        fate_entries.append((lt, a[i], sub_ct[lt], "sub"))
                        break
                else:
                    continue
                break
        if drop_ct[lt]:
            fate_entries.append((lt, None, drop_ct[lt], "drop"))

    for t in all_tools:
        if ins_ct[t]:
            fate_entries.append((None, t, ins_ct[t], "ins"))

    # Draw Sankey flow arcs
    for entry in fate_entries:
        lt, rt, w, fate = entry
        color = COLORS[fate]
        lw = 1.5 + w * 2.5
        ly = tool_y[lt] if lt else -0.05
        ry = tool_y[rt] if rt else 1.05

        ax.annotate(
            "",
            xy=(X_RIGHT, ry),
            xycoords="data",
            xytext=(X_LEFT, ly),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                connectionstyle="arc3,rad=0.0",
                mutation_scale=12,
                alpha=0.65,
            ),
            zorder=2,
        )

    # Right fate labels
    for fate, label in [
        ("kept", "Kept"),
        ("sub", "Substituted"),
        ("drop", "Dropped"),
        ("ins", "Inserted"),
    ]:
        total = sum(w for _, _, w, f in fate_entries if f == fate)
        if total:
            ys_fate = [tool_y[rt] for _, rt, _, f in fate_entries if f == fate and rt]
            yc = np.mean(ys_fate) if ys_fate else 0.5
            ax.text(
                X_RIGHT + 0.02,
                yc,
                f"{label}\n(n={total})",
                fontsize=8.5,
                color=COLORS[fate],
                fontweight="bold",
                va="center",
            )

    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    ax.set_title(
        "Sankey flow: benign tool fate under PGD-L∞ attack",
        fontsize=14,
        fontweight="bold",
        color=C_NODE,
        pad=15,
    )
    ax.text(
        X_LEFT - 0.05,
        0.97,
        "Benign tools",
        ha="center",
        fontsize=10,
        color=C_BENIGN,
        fontweight="bold",
    )
    ax.text(X_RIGHT + 0.02, 0.97, "Fate", ha="left", fontsize=10, color=C_NODE, fontweight="bold")

    fig.savefig(OUT / "graph4_sankey.png")
    plt.close(fig)
    print("graph4 ✓")
