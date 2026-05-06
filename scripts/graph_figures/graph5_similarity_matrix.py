"""Graph 5: pairwise trajectory edit-distance heatmap (all conditions)."""

from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from ._common import C_NODE, OUT, _load


def graph5_similarity_matrix() -> None:
    from adversarial_reasoning.metrics.trajectory import trajectory_edit_distance

    pgd_recs = _load("runs/pgd_smoke/records.jsonl")
    noise_recs = {r["sample_id"]: r for r in _load("runs/smoke/records.jsonl")}

    # Build (label, sequence) list per patient
    [r["sample_id"].split("_p")[1] for r in pgd_recs]
    all_seqs: list[tuple[str, list[str]]] = []
    all_labels: list[str] = []

    for r in pgd_recs:
        pid = r["sample_id"].split("_p")[1]
        nr = noise_recs.get(r["sample_id"])
        all_seqs.append((f"B·{pid}", r["benign"]["tool_sequence"]))
        all_seqs.append((f"N·{pid}", nr["attacked"]["tool_sequence"] if nr else []))
        all_seqs.append((f"P·{pid}", r["attacked"]["tool_sequence"]))
        all_labels += [f"B·{pid}", f"N·{pid}", f"P·{pid}"]

    n = len(all_seqs)
    mat = np.zeros((n, n))
    for i, (_, si) in enumerate(all_seqs):
        for j, (_, sj) in enumerate(all_seqs):
            mat[i, j] = trajectory_edit_distance(si, sj, normalize=True)

    # Cluster by condition groups
    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = cm.get_cmap("magma")
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    labels = [lbl for lbl, _ in all_seqs]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            tcolor = "white" if v > 0.5 else "#cccccc"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=tcolor)

    # Draw group separators every 3 rows/cols
    for k in range(1, len(pgd_recs)):
        ax.axhline(k * 3 - 0.5, color="#30363d", lw=1.5)
        ax.axvline(k * 3 - 0.5, color="#30363d", lw=1.5)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Normalised edit distance", fontsize=10, color=C_NODE)
    cb.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

    # Legend for condition codes
    legend_txt = "B = benign  ·  N = noise-attacked  ·  P = PGD-attacked"
    ax.set_title(
        f"Pairwise trajectory edit-distance matrix (all conditions, n=5 patients)\n{legend_txt}",
        fontsize=11,
        fontweight="bold",
        color=C_NODE,
        pad=12,
    )
    fig.savefig(OUT / "graph5_similarity_matrix.png")
    plt.close(fig)
    print("graph5 ✓")
