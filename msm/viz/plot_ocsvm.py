"""Visualization for OneClassSVM OOD detection results.

Replaces plot_gmm.py — all references to GMM (score_samples, means_, n_components,
log-likelihood) are replaced with OCSVM equivalents (decision_function, boundary=0).

The natural OCSVM decision boundary is at 0:
  decision > 0  →  inlier
  decision < 0  →  OOD

Plots produced
--------------
plot_ocsvm_for_all_frauds()
    • ocsvm_all_frauds_{level}.svg/png   — all frauds overlaid on decision surface
    • ocsvm_origins_vs_{fraud}_{level}.svg — one plot per fraud type

plot_ocsvm_grouped()
    • ocsvm_grouped_{level}.svg/png      — attack groups on decision surface
    • ocsvm_origins_vs_{group}_{level}.svg — one plot per attack group

plot_auroc_bar_grouped()
    • auroc_grouped_{level}.svg/png      — AUROC bar chart per group
"""

import numpy as np
import matplotlib.pyplot as plt
import cycler

# Enable grid and update its appearance
plt.rcParams.update({'axes.grid': True})
plt.rcParams.update({'grid.color': 'silver'})
plt.rcParams.update({'grid.linestyle': '--'})

# Set figure resolution
plt.rcParams.update({'figure.dpi': 150})

# Hide the top and right spines
plt.rcParams.update({'axes.spines.top': False})
plt.rcParams.update({'axes.spines.right': False})

# Increase font sizes
plt.rcParams.update({'font.size': 12})  # General font size
plt.rcParams.update({'axes.titlesize': 14})  # Title font size
plt.rcParams.update({'axes.labelsize': 12})  # Axis label font size

plt.rcParams.update({'axes.prop_cycle': cycler.cycler('color', ['#fd8e26'])})
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, Any

from eval.grouping import FRAUD_GROUPS


# ── Color scheme ──────────────────────────────────────────────────────────────
GROUP_COLORS = {
    "origins":                 "#2ca02c",   # green
    "static_midvholo":         "#1f77b4",   # blue
    "static_midvdynattack":    "#ff7f0e",   # orange
    "dynamic_midvdynattack":   "#d62728",   # red
    "swap_midvdynattack":      "#9467bd",   # purple
    "photo_replacement":       "#8c564b",   # brown
}


def _short_name(name: str, max_len: int = 22) -> str:
    """Truncate long fraud names for axis labels."""
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def _to_xy(d: Dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([d["recon_cosine"], d["seq_cosine"]]).T


def _subsample(xy: np.ndarray, max_points: int, seed: int = 42) -> np.ndarray:
    if len(xy) > max_points:
        idx = np.random.default_rng(seed).choice(len(xy), max_points, replace=False)
        return xy[idx]
    return xy


def _build_grid(
    all_xy: np.ndarray,
    scaler,
    model,
    resolution: int = 220,
):
    """Build meshgrid and evaluate OCSVM decision function over it.

    Returns
    -------
    xx, yy : meshgrid arrays (original space)
    Z      : decision_function values reshaped to (resolution, resolution)
    x_min, x_max, y_min, y_max : plot bounds
    """
    margin = 0.06 * (all_xy.max(axis=0) - all_xy.min(axis=0))
    x_min, x_max = all_xy[:, 0].min() - margin[0], all_xy[:, 0].max() + margin[0]
    y_min, y_max = all_xy[:, 1].min() - margin[1], all_xy[:, 1].max() + margin[1]

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid_points)
    Z = model.decision_function(grid_scaled).reshape(xx.shape)

    return xx, yy, Z, x_min, x_max, y_min, y_max


def _draw_decision_surface(ax, xx, yy, Z, thresholds):
    """Draw the OCSVM decision surface, boundary, and p95/p99 contours."""
    # Background: continuous decision score
    levels_fill = np.linspace(Z.min(), Z.max(), 30)
    cf = ax.contourf(xx, yy, Z, levels=levels_fill, cmap="RdYlGn", alpha=0.45)

    # Natural OCSVM boundary (decision = 0) — solid white
    ax.contour(xx, yy, Z, levels=[0.0], colors="white",
               linewidths=2.5, linestyles="-")

    # p95 threshold (anom = p95  →  decision = -p95)
    p95_decision = -thresholds["p95"]
    p99_decision = -thresholds["p99"]
    ax.contour(xx, yy, Z, levels=[p95_decision], colors="orange",
               linewidths=2.0, linestyles="--")
    ax.contour(xx, yy, Z, levels=[p99_decision], colors="red",
               linewidths=2.0, linestyles=":")

    return cf


def _threshold_legend_handles(thresholds):
    """Return proxy legend handles for the three threshold contours."""
    val_ood = thresholds.get("val_ood_rate", 0.0)
    return [
        Line2D([0], [0], color="white",  lw=2.5, ls="-",
               label=f"OCSVM boundary (val OOD={val_ood:.1%})"),
        Line2D([0], [0], color="orange", lw=2.0, ls="--",
               label="p95 threshold"),
        Line2D([0], [0], color="red",    lw=2.0, ls=":",
               label="p99 threshold"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def plot_ocsvm_for_all_frauds(
    ocsvm_results: Dict[str, Any],
    id_scores: Dict[str, np.ndarray],
    test_scores_map: Dict[str, Dict[str, np.ndarray]],
    level_name: str = "sequence",
    out_dir: str | Path | None = None,
    max_points: int = 500,
):
    """Plot OCSVM decision surface with fraud overlays.

    Parameters
    ----------
    ocsvm_results   : return value of fit_ocsvm_and_score()
    id_scores       : validation / in-distribution scores used for fitting
    test_scores_map : mapping name -> {signal -> array} for all test sets
    level_name      : 'sequence' or 'video'
    out_dir         : directory to save plots (skipped if None)
    max_points      : subsample cap per class for scatter points
    """
    model      = ocsvm_results["model"]
    scaler     = ocsvm_results["scaler"]
    thresholds = ocsvm_results["thresholds"]

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    origins_xy = _to_xy(id_scores)

    # Collect all points for grid bounds
    all_xy = np.vstack([origins_xy] + [_to_xy(v) for v in test_scores_map.values()])
    xx, yy, Z, *_ = _build_grid(all_xy, scaler, model)

    # ──────────────────────────────────────────────────────────
    # PLOT 1: all frauds together
    # ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 9))

    cf = _draw_decision_surface(ax, xx, yy, Z, thresholds)
    plt.colorbar(cf, ax=ax, label="Decision score (+ = inlier, − = OOD)", shrink=0.8)

    # Origins
    o_xy = _subsample(origins_xy, max_points)
    ax.scatter(o_xy[:, 0], o_xy[:, 1], c="#2ca02c", s=22, alpha=0.55,
               marker="o", edgecolors="none", label="Origins (ID)", zorder=5)

    # All frauds
    cmap = plt.cm.get_cmap("tab20", len(test_scores_map))
    for i, (name, sdict) in enumerate(test_scores_map.items()):
        if name == "origins":
            continue
        xy = _subsample(_to_xy(sdict), max_points, seed=42 + i)
        ax.scatter(xy[:, 0], xy[:, 1], c=[cmap(i)], s=14, alpha=0.6,
                   marker="x", linewidths=0.8,
                   label=_short_name(name, 22), zorder=6)

    threshold_handles = _threshold_legend_handles(thresholds)
    legend1 = ax.legend(handles=threshold_handles, loc="lower right", fontsize=8,
                        title="Boundaries", framealpha=0.85)
    ax.add_artist(legend1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.18, 1.0), fontsize=7,
              title="Classes", framealpha=0.85)

    nu_str = f"nu={ocsvm_results['nu']}  kernel={ocsvm_results['kernel']}"
    ax.set_xlabel("Reconstruction Cosine Distance", fontsize=11)
    ax.set_ylabel("Sequence Cosine Distance", fontsize=11)
    ax.set_title(
        f"OneClassSVM — All Frauds ({level_name} level)"
        f"{nu_str}  |  White=boundary  Orange=p95  Red=p99",
        fontsize=12, fontweight="bold",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if out_dir:
        stem = f"ocsvm_all_frauds_{level_name}"
        fig.savefig(Path(out_dir) / f"{stem}.svg", format="svg", bbox_inches="tight")
        fig.savefig(Path(out_dir) / f"{stem}.png", dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved {stem}.svg/png")
    plt.close(fig)

    # ──────────────────────────────────────────────────────────
    # PLOT 2: one plot per fraud type
    # ──────────────────────────────────────────────────────────
    fraud_names = [n for n in test_scores_map if n != "origins"]

    for fraud_name in fraud_names:
        fig, ax = plt.subplots(figsize=(10, 8))

        cf = _draw_decision_surface(ax, xx, yy, Z, thresholds)
        plt.colorbar(cf, ax=ax, label="Decision score", shrink=0.75)

        # Origins
        ax.scatter(o_xy[:, 0], o_xy[:, 1], c="#2ca02c", s=28, alpha=0.55,
                   marker="o", edgecolors="none", label="Origins (ID)", zorder=5)

        # Single fraud
        fraud_xy = _subsample(_to_xy(test_scores_map[fraud_name]), max_points, seed=1)
        ax.scatter(fraud_xy[:, 0], fraud_xy[:, 1], c="#d62728", s=22, alpha=0.7,
                   marker="x", linewidths=0.9, label=fraud_name, zorder=6)

        # Stats annotation
        ood_bd  = ocsvm_results["ood_rates"].get(fraud_name, {}).get("boundary", 0)
        ood_p95 = ocsvm_results["ood_rates"].get(fraud_name, {}).get("p95", 0)
        ood_p99 = ocsvm_results["ood_rates"].get(fraud_name, {}).get("p99", 0)
        auroc_v = ocsvm_results["auroc"].get(fraud_name, 0)
        stats_text = (
            f"AUROC: {auroc_v:.3f}",
            f"OOD@boundary: {ood_bd:.1%}",
            f"OOD@p95: {ood_p95:.1%}",
            f"OOD@p99: {ood_p99:.1%}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        threshold_handles = _threshold_legend_handles(thresholds)
        legend1 = ax.legend(handles=threshold_handles, loc="lower right", fontsize=8,
                            framealpha=0.85)
        ax.add_artist(legend1)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)

        ax.set_xlabel("Reconstruction Cosine Distance", fontsize=11)
        ax.set_ylabel("Sequence Cosine Distance", fontsize=11)
        ax.set_title(
            f"Origins vs {fraud_name} ({level_name} level) "
            f"OneClassSVM  nu={ocsvm_results['nu']}",
            fontsize=12, fontweight="bold",
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()

        if out_dir:
            safe = fraud_name.replace(" ", "_")
            path = Path(out_dir) / f"ocsvm_origins_vs_{safe}_{level_name}.svg"
            fig.savefig(path, format="svg", bbox_inches="tight")
            print(f"  ✓ Saved ocsvm_origins_vs_{safe}_{level_name}.svg")
        plt.close(fig)


def plot_ocsvm_grouped(
    ocsvm_results: Dict[str, Any],
    grouped_scores: Dict[str, Dict[str, np.ndarray]],
    origins_scores: Dict[str, np.ndarray],
    stats: Dict[str, Any],
    level_name: str = "sequence",
    out_dir: str | Path | None = None,
    max_points: int = 500,
):
    """Plot OCSVM decision surface with attack groups color-coded.

    Parameters
    ----------
    ocsvm_results   : return value of fit_ocsvm_and_score()
    grouped_scores  : mapping group_name -> {signal -> array}
    origins_scores  : in-distribution scores dict
    stats           : output of compute_stats_all_levels()
    level_name      : 'sequence' or 'video'
    out_dir         : directory to save plots
    max_points      : subsample cap per class
    """
    model      = ocsvm_results["model"]
    scaler     = ocsvm_results["scaler"]
    thresholds = ocsvm_results["thresholds"]

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    origins_xy = _to_xy(origins_scores)
    all_xy = np.vstack(
        [origins_xy] + [_to_xy(v) for v in grouped_scores.values()]
    )
    xx, yy, Z, *_ = _build_grid(all_xy, scaler, model)
    o_xy = _subsample(origins_xy, max_points)

    markers = ["s", "^", "D", "v", "P"]

    # ──────────────────────────────────────────────────────────
    # PLOT: all groups together
    # ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    cf = _draw_decision_surface(ax, xx, yy, Z, thresholds)
    plt.colorbar(cf, ax=ax, label="Decision score (+ = inlier, − = OOD)", shrink=0.8)

    ax.scatter(o_xy[:, 0], o_xy[:, 1], c=GROUP_COLORS["origins"],
               s=25, alpha=0.5, marker="o", edgecolors="none",
               label="Origins (ID)", zorder=5)

    for i, (group_name, _) in enumerate(FRAUD_GROUPS.items()):
        if group_name not in grouped_scores:
            continue
        xy = _subsample(_to_xy(grouped_scores[group_name]), max_points, seed=42 + i)
        m  = stats["grouped"].get(group_name, {})
        color  = GROUP_COLORS.get(group_name, f"C{i}")
        marker = markers[i % len(markers)]
        label  = (
            f"{group_name}"
            f"(n={m.get('n_samples', len(xy))}, "
            f"AUROC={m.get('auroc', 0):.2f}, "
            f"OOD@bd={m.get('ood_boundary', 0):.0%})"
        )
        ax.scatter(xy[:, 0], xy[:, 1], c=color, s=28, alpha=0.6,
                   marker=marker, label=label, zorder=6)

    threshold_handles = _threshold_legend_handles(thresholds)
    legend1 = ax.legend(handles=threshold_handles, loc="lower right", fontsize=8,
                        title="Boundaries", framealpha=0.85)
    ax.add_artist(legend1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9,
              title="Attack Groups")

    nu_str = f"nu={ocsvm_results['nu']}  kernel={ocsvm_results['kernel']}"
    ax.set_xlabel("Reconstruction Cosine Distance", fontsize=12)
    ax.set_ylabel("Sequence Cosine Distance", fontsize=12)
    ax.set_title(
        f"OneClassSVM — Attack Groups ({level_name} level)"
        f"{nu_str}  |  White=boundary  Orange=p95  Red=p99",
        fontsize=14, fontweight="bold",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if out_dir:
        stem = f"ocsvm_grouped_{level_name}"
        fig.savefig(Path(out_dir) / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(Path(out_dir) / f"{stem}.png", dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved {stem}.svg/png")
    plt.close(fig)

    # ──────────────────────────────────────────────────────────
    # PLOT: one per group
    # ──────────────────────────────────────────────────────────
    for group_name in FRAUD_GROUPS.keys():
        if group_name not in grouped_scores:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        cf = _draw_decision_surface(ax, xx, yy, Z, thresholds)
        plt.colorbar(cf, ax=ax, label="Decision score", shrink=0.75)

        ax.scatter(o_xy[:, 0], o_xy[:, 1], c=GROUP_COLORS["origins"],
                   s=28, alpha=0.5, marker="o", edgecolors="none",
                   label="Origins (ID)", zorder=5)

        xy = _subsample(_to_xy(grouped_scores[group_name]), max_points * 2)
        color = GROUP_COLORS.get(group_name, "red")
        ax.scatter(xy[:, 0], xy[:, 1], c=color, s=28, alpha=0.65,
                   marker="x", linewidths=0.9, label=group_name, zorder=6)

        # Stats box
        m = stats["grouped"].get(group_name, {})
        stats_text = (
            f"N = {m.get('n_samples', 0)}"
            f"AUROC = {m.get('auroc', 0):.3f}"
            f"OOD@boundary = {m.get('ood_boundary', 0):.1%}"
            f"OOD@p95 = {m.get('ood_p95', 0):.1%}"
            f"OOD@p99 = {m.get('ood_p99', 0):.1%}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

        threshold_handles = _threshold_legend_handles(thresholds)
        legend1 = ax.legend(handles=threshold_handles, loc="lower right", fontsize=8,
                            framealpha=0.85)
        ax.add_artist(legend1)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)

        ax.set_xlabel("Reconstruction Cosine Distance", fontsize=11)
        ax.set_ylabel("Sequence Cosine Distance", fontsize=11)
        ax.set_title(
            f"Origins vs {group_name} ({level_name} level)"
            f"OneClassSVM  nu={ocsvm_results['nu']}",
            fontsize=13, fontweight="bold",
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()

        if out_dir:
            path = Path(out_dir) / f"ocsvm_origins_vs_{group_name}_{level_name}.svg"
            fig.savefig(path, bbox_inches="tight")
            print(f"  ✓ Saved ocsvm_origins_vs_{group_name}_{level_name}.svg")
        plt.close(fig)


def plot_auroc_bar_grouped(
    stats: Dict[str, Any],
    level_name: str = "sequence",
    out_dir: str | Path | None = None,
):
    """Bar chart comparing AUROC across attack groups."""
    groups  = [g for g in FRAUD_GROUPS.keys() if g in stats["grouped"]]
    aurocs  = [stats["grouped"][g]["auroc"]    for g in groups]
    colors  = [GROUP_COLORS.get(g, "gray")     for g in groups]
    samples = [stats["grouped"][g]["n_samples"] for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(groups)), aurocs, color=colors,
                  edgecolor="black", linewidth=0.6)

    for bar, n in zip(bars, samples):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.5, color="gray",  linestyle=":", linewidth=1, alpha=0.7, label="Random")
    ax.axhline(0.9, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Target (0.9)")
    ax.set_title(f"AUROC by Attack Group — OneClassSVM ({level_name} level)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        stem = f"auroc_grouped_{level_name}"
        fig.savefig(Path(out_dir) / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(Path(out_dir) / f"{stem}.png", dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved {stem}.svg/png")
    plt.close(fig)


def plot_ocsvm_val_vs_test(
    ocsvm_results: Dict[str, Any],
    id_scores: Dict[str, np.ndarray],           # val legits (left panel)
    origins_test_scores: Dict[str, np.ndarray], # test legits (right panel, "origins" key)
    grouped_scores: Dict[str, Dict[str, np.ndarray]],  # grouped test frauds (right panel)
    level_name: str = "sequence",
    out_dir: str | Path | None = None,
    max_points: int = 400,
):
    """Two-panel figure: val legits only (left) | test legits + frauds (right).

    Left  — decision surface + OCSVM boundary (white solid)
    Right — same surface + boundary + p95 (orange dashed) + p99 (red dotted)
              + test legits + grouped frauds
    """
    model      = ocsvm_results["model"]
    scaler     = ocsvm_results["scaler"]
    thresholds = ocsvm_results["thresholds"]

    # ── Build common grid from ALL points so both panels share the same axes ──
    val_xy   = _to_xy(id_scores)
    orig_xy  = _to_xy(origins_test_scores)
    fraud_xys = [_to_xy(v) for v in grouped_scores.values()]
    all_xy   = np.vstack([val_xy, orig_xy] + fraud_xys)
    xx, yy, Z, x_min, x_max, y_min, y_max = _build_grid(all_xy, scaler, model)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    # ── Shared colorbar setup ──────────────────────────────────────────────────
    vmin, vmax = Z.min(), Z.max()
    levels_fill = np.linspace(vmin, vmax, 30)

    for ax in axes:
        ax.contourf(xx, yy, Z, levels=levels_fill, cmap="RdYlGn", alpha=0.45,
                    vmin=vmin, vmax=vmax)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Sequence reconstructability\n"
                      "(mean cos. dist. between MSM in/out embeddings)", fontsize=10)

    axes[0].set_ylabel("Sequence similarity\n"
                       "(mean projector embeddings pairwise cos. dist.)", fontsize=10)

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT PANEL — Validation Set (Legits Only)
    # ══════════════════════════════════════════════════════════════════════════
    ax_l = axes[0]

    # Decision surface boundary only (no p95/p99 on val panel)
    ax_l.contour(xx, yy, Z, levels=[0.0], colors="white",
                 linewidths=2.5, linestyles="-")

    # Legit scatter
    v_xy = _subsample(val_xy, max_points)
    ax_l.scatter(v_xy[:, 0], v_xy[:, 1], c="#2ca02c", s=22, alpha=0.55,
                 marker="o", edgecolors="none", label="Legits (val)", zorder=5)

    ax_l.set_title("Validation Set (Legits Only)\nOCSVM decision boundary", fontsize=12, fontweight="bold")

    ax_l.legend(
        handles=[Line2D([0], [0], color="white", lw=2.5, ls="-", label="OCSVM boundary")],
        loc="lower right", fontsize=9, framealpha=0.85
    )

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT PANEL — Test Set (Legits & Attacks)
    # ══════════════════════════════════════════════════════════════════════════
    ax_r = axes[1]

    # All three contours
    ax_r.contour(xx, yy, Z, levels=[0.0],                  colors="white",  linewidths=2.5, linestyles="-")
    ax_r.contour(xx, yy, Z, levels=[-thresholds["p95"]],   colors="orange", linewidths=2.0, linestyles="--")
    ax_r.contour(xx, yy, Z, levels=[-thresholds["p99"]],   colors="red",    linewidths=2.0, linestyles=":")

    # Test legits (origins)
    o_xy = _subsample(orig_xy, max_points)
    ax_r.scatter(o_xy[:, 0], o_xy[:, 1], c="#2ca02c", s=22, alpha=0.55,
                 marker="o", edgecolors="none", label="Legits (test)", zorder=5)

    # Frauds per group
    markers = ["x", "^", "s", "D", "P", "v"]
    for i, (group_name, _) in enumerate(FRAUD_GROUPS.items()):
        if group_name not in grouped_scores:
            continue
        xy = _subsample(_to_xy(grouped_scores[group_name]), max_points, seed=42 + i)
        # auroc_v = ocsvm_results.get("auroc", {}).get(group_name, float("nan"))
        color   = GROUP_COLORS.get(group_name, f"C{i}")
        auroc_v = ocsvm_results.get("auroc", {}).get(group_name, float("nan"))
        if np.isnan(auroc_v) and group_name in grouped_scores:
            from eval.ocsvm import auroc as _auroc
            id_xy   = _to_xy(origins_test_scores)          # test legits as the "ID" reference
            fr_xy   = _to_xy(grouped_scores[group_name])
            scaler_ = ocsvm_results["scaler"]
            model_  = ocsvm_results["model"]
            id_dec  = model_.decision_function(scaler_.transform(id_xy))
            fr_dec  = model_.decision_function(scaler_.transform(fr_xy))
            auroc_v = 1-_auroc(id_dec, fr_dec)
        label = f"{group_name}  (AUROC={auroc_v:.2f})"
        label   = f"{group_name}  (AUROC={auroc_v:.2f})"
        ax_r.scatter(xy[:, 0], xy[:, 1], c=color, s=28, alpha=0.65,
                     marker=markers[i % len(markers)], linewidths=0.9,
                     label=label, zorder=6)

    ax_r.set_title("Test Set (Legits & Attacks)\nLikelihood frontiers", fontsize=12, fontweight="bold")

    # Boundary legend (lower right)
    boundary_handles = [
        Line2D([0], [0], color="white",  lw=2.5, ls="-",  label="OCSVM boundary"),
        Line2D([0], [0], color="orange", lw=2.0, ls="--", label=f"OOD@p95  (val thr={thresholds['p95']:.3f})"),
        Line2D([0], [0], color="red",    lw=2.0, ls=":",  label=f"OOD@p99  (val thr={thresholds['p99']:.3f})"),
    ]
    leg1 = ax_r.legend(handles=boundary_handles, loc="lower right", fontsize=8,
                       title="Boundaries (val-calibrated)", framealpha=0.85)
    ax_r.add_artist(leg1)
    ax_r.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
                title="Classes", framealpha=0.85)

    # # ── Arrow annotation between panels ───────────────────────────────────────
    # fig.text(0.505, 0.52, "→", fontsize=36, ha="center", va="center",
    #          color="#1a5276", fontweight="bold")

    fig.suptitle(
        f"OneClassSVM OOD Detection  —  {level_name} level  |  nu={ocsvm_results['nu']}",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        stem = f"ocsvm_val_vs_test_{level_name}"
        fig.savefig(Path(out_dir) / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(Path(out_dir) / f"{stem}.png", dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved {stem}.svg/png")
    plt.close(fig)
