"""
Report figures for the keystroke verification pipeline.
Expects the summary dict returned by keystroke.run_dataset_demo().
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def _apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})


def save_report_figures(summary: dict, out_dir: Path) -> List[Path]:
    """
    Write numbered PNGs into out_dir. Returns paths in display order.
    Required summary keys: model, genuine_subject, threshold, genuine_subspace_distances,
    impostor_subspace_distances, train_subspace_distances, genuine_accept, genuine_total,
    impostor_reject, impostor_total, n_features, train_n.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    vm = summary["model"]
    thr = float(summary["threshold"])
    subj = summary["genuine_subject"]
    evals = vm.pca.eigenvalues
    ratios = vm.pca.explained_variance_ratio
    k = len(evals)
    g_dist = np.asarray(summary["genuine_subspace_distances"])
    i_dist = np.asarray(summary["impostor_subspace_distances"])
    t_dist = np.asarray(summary["train_subspace_distances"])

    saved: List[Path] = []

    # 01 — PCA eigenvalues
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(np.arange(1, k + 1), evals, "o-", color="#1565c0", ms=7, lw=2, clip_on=False)
    ax.set_xlabel("Component index (largest eigenvalues first)")
    ax.set_ylabel("Eigenvalue λ")
    ax.set_title("Spectrum of covariance matrix after scaling and centering")
    fig.tight_layout()
    p1 = out_dir / "01_pca_eigenvalues.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p1)

    # 02 — Explained variance share
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(1, k + 1)
    ax.bar(x, ratios * 100.0, color="#2e7d32", edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Share of total variance (%)")
    ax.set_title("Variance captured by each retained component")
    fig.tight_layout()
    p2 = out_dir / "02_variance_explained.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p2)

    # 03 — Genuine vs impostor distance distributions
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(
        g_dist,
        bins=28,
        alpha=0.75,
        color="#1565c0",
        label=f"Genuine holdout (n={len(g_dist)})",
        density=True,
    )
    ax.hist(
        i_dist,
        bins=28,
        alpha=0.7,
        color="#c62828",
        label=f"Impostor sample (n={len(i_dist)})",
        density=True,
    )
    ax.axvline(thr, color="black", lw=2.2, ls="--", label=f"Threshold τ = {thr:.3f}")
    ax.set_xlabel(r"Subspace score $\|P^T(x_{scaled} - \mu)\|$")
    ax.set_ylabel("Density")
    ax.set_title(f"Verification scores — enrolled subject {subj}")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    p3 = out_dir / "03_genuine_vs_impostor_distances.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p3)

    # 04 — Training set distances vs threshold
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(t_dist, bins=32, color="#5e35b1", alpha=0.88, edgecolor="white", linewidth=0.6)
    ax.axvline(thr, color="black", lw=2.2, ls="--", label=f"Threshold τ = {thr:.3f}")
    ax.set_xlabel(r"Training score $\|P^T(x - \mu)\|$")
    ax.set_ylabel("Count")
    ax.set_title("Enrollment (training) samples vs decision threshold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    p4 = out_dir / "04_training_distances.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p4)

    # 05 — Pipeline schematic
    fig, ax = plt.subplots(figsize=(9.0, 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = [
        "Feature\nvector x",
        "Diagonal\nscaling",
        "Center\n& PCA",
        "Project\nto ℝᵏ",
        "Euclidean\ndistance",
        "Compare\nto τ",
    ]
    nbox = len(labels)
    xs = np.linspace(0.06, 0.94, nbox)
    for i, (x, lab) in enumerate(zip(xs, labels)):
        w, h = 0.11, 0.38
        bx = FancyBboxPatch(
            (x - w / 2, 0.5 - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="#e8eaf6",
            edgecolor="#3949ab",
            linewidth=1.4,
        )
        ax.add_patch(bx)
        ax.text(x, 0.5, lab, ha="center", va="center", fontsize=9, fontweight="bold", color="#1a237e")
        if i < nbox - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - w / 2 - 0.01, 0.5),
                xytext=(x + w / 2 + 0.01, 0.5),
                arrowprops=dict(arrowstyle="->", color="#424242", lw=1.6, shrinkA=0, shrinkB=0),
            )
    ax.text(0.5, 0.92, "Verification pipeline", ha="center", fontsize=12, fontweight="bold", color="#212121")
    fig.tight_layout()
    p5 = out_dir / "05_pipeline_schematic.png"
    fig.savefig(p5, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p5)

    # 06 — Classification rates
    g_tot = max(int(summary["genuine_total"]), 1)
    i_tot = max(int(summary["impostor_total"]), 1)
    g_rate = 100.0 * float(summary["genuine_accept"]) / g_tot
    i_rate = 100.0 * float(summary["impostor_reject"]) / i_tot
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    cats = ["Genuine\nholdout", "Impostor\nholdout"]
    vals = [g_rate, i_rate]
    colors = ["#1565c0", "#c62828"]
    bars = ax.bar(cats, vals, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
    ax.set_ylabel("Correct decision (%)")
    ax.set_title("Holdout classification accuracy")
    ax.set_ylim(0, 105)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{v:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    p6 = out_dir / "06_holdout_accuracy.png"
    fig.savefig(p6, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p6)

    # 07 — Feature space summary (text panel)
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    ax.axis("off")
    lines = [
        f"Dataset rows (genuine train): {summary['train_n']}",
        f"Feature dimension d: {summary['n_features']}",
        f"PCA subspace dimension k: {k}",
        f"Enrolled subject id: {subj}",
        f"Decision threshold τ: {thr:.4f}",
    ]
    ax.text(0.02, 0.95, "Experiment summary", fontsize=12, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.02, 0.72, "\n".join(lines), fontsize=10, va="top", family="monospace", transform=ax.transAxes)
    fig.tight_layout()
    p7 = out_dir / "07_summary_panel.png"
    fig.savefig(p7, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(p7)

    return saved
