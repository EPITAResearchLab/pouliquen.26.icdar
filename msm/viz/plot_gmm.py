import numpy as np
from pathlib import Path


def _short_name(name: str, max_len: int = 18) -> str:
    """Truncate long fraud names for axis labels."""
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def plot_gmm_for_all_frauds(
    gmm_results: dict,
    id_scores: dict[str, np.ndarray],
    test_scores_map: dict[str, dict[str, np.ndarray]],
    level_name: str = "sequence",
    out_dir: str | Path | None = None,
    max_points: int = 500,
):
    """Plot GMM decision boundaries with each fraud type separately.
    
    Parameters
    ----------
    gmm_results : output from fit_gmm_and_score()
    id_scores   : validation/origins scores (used for fitting)
    test_scores_map : mapping name -> {signal -> array} for all test sets
    level_name  : 'sequence' or 'video'
    out_dir     : directory to save plots
    max_points  : subsample cap per class
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    
    gmm = gmm_results["gmm"]
    scaler = gmm_results["scaler"]
    thresholds = gmm_results["thresholds"]
    
    def _to_xy(d):
        x = d["recon_cosine"]
        y = d["seq_cosine"]
        return np.vstack([x, y]).T
    
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Get origins data
    origins_xy = _to_xy(id_scores)
    
    # Collect all points to determine plot bounds
    all_xy = [origins_xy]
    for sdict in test_scores_map.values():
        all_xy.append(_to_xy(sdict))
    all_xy = np.vstack(all_xy)
    
    margin = 0.05 * (all_xy.max(axis=0) - all_xy.min(axis=0))
    x_min, x_max = all_xy[:, 0].min() - margin[0], all_xy[:, 0].max() + margin[0]
    y_min, y_max = all_xy[:, 1].min() - margin[1], all_xy[:, 1].max() + margin[1]
    
    # Create mesh grid for contours
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid_points)
    log_probs = gmm.score_samples(grid_scaled)
    Z = log_probs.reshape(xx.shape)
    
    # Convert thresholds to log-likelihood (threshold is on -loglik)
    p95_loglik = -thresholds["p95"]
    p99_loglik = -thresholds["p99"]
    
    # ═══════════════════════════════════════════════════════════
    # PLOT 1: All frauds together
    # ═══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Density contours
    levels = np.linspace(Z.min(), Z.max(), 25)
    contour = ax.contourf(xx, yy, Z, levels=levels, cmap='viridis', alpha=0.5)
    plt.colorbar(contour, ax=ax, label='Log-likelihood', shrink=0.8)
    
    # Threshold contours
    ax.contour(xx, yy, Z, levels=[p99_loglik], colors='red', 
               linewidths=2, linestyles='--')
    ax.contour(xx, yy, Z, levels=[p95_loglik], colors='orange', 
               linewidths=2, linestyles='-')
    
    # Origins
    o_xy = origins_xy
    if len(o_xy) > max_points:
        idx = np.random.default_rng(42).choice(len(o_xy), max_points, replace=False)
        o_xy = o_xy[idx]
    ax.scatter(o_xy[:, 0], o_xy[:, 1], c='#2ca02c', s=20, alpha=0.5, 
               marker='o', label='Origins (ID)', zorder=5)
    
    # All frauds
    cmap = plt.cm.get_cmap("tab20", len(test_scores_map))
    for i, (name, sdict) in enumerate(test_scores_map.items()):
        if name == "origins":
            continue
        xy = _to_xy(sdict)
        if len(xy) > max_points:
            idx = np.random.default_rng(42).choice(len(xy), max_points, replace=False)
            xy = xy[idx]
        ax.scatter(xy[:, 0], xy[:, 1], c=[cmap(i)], s=15, alpha=0.6,
                   marker='x', label=_short_name(name, 20), zorder=6)
    
    # GMM centers (in original space)
    means_original = scaler.inverse_transform(gmm.means_)
    ax.scatter(means_original[:, 0], means_original[:, 1], 
               c='red', s=300, marker='*', edgecolors='black', 
               linewidths=2, label='GMM Centers', zorder=10)
    
    ax.set_xlabel('Reconstruction Cosine Distance', fontsize=11)
    ax.set_ylabel('Sequence Cosine Distance', fontsize=11)
    ax.set_title(f'GMM Anomaly Detection — All Frauds ({level_name} level)\n'
                 f'Components: {gmm.n_components} | Orange=p95 | Red=p99',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.0), fontsize=7)
    ax.grid(alpha=0.2)
    
    fig.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / f"gmm_all_frauds_{level_name}.svg", 
                    format='svg', bbox_inches='tight')
        fig.savefig(Path(out_dir) / f"gmm_all_frauds_{level_name}.png", 
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved gmm_all_frauds_{level_name}.svg/png")
    
    # ═══════════════════════════════════════════════════════════
    # PLOT 2: One plot per fraud type (origins vs single fraud)
    # ═══════════════════════════════════════════════════════════
    fraud_names = [n for n in test_scores_map if n != "origins"]
    
    for fraud_name in fraud_names:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Density contours
        contour = ax.contourf(xx, yy, Z, levels=levels, cmap='viridis', alpha=0.4)
        plt.colorbar(contour, ax=ax, label='Log-likelihood', shrink=0.7)
        
        # Threshold contours
        c95 = ax.contour(xx, yy, Z, levels=[p95_loglik], colors='orange', 
                         linewidths=2.5, linestyles='-')
        c99 = ax.contour(xx, yy, Z, levels=[p99_loglik], colors='red', 
                         linewidths=2.5, linestyles='--')
        
        # Origins
        o_xy = origins_xy
        if len(o_xy) > max_points:
            idx = np.random.default_rng(42).choice(len(o_xy), max_points, replace=False)
            o_xy = o_xy[idx]
        ax.scatter(o_xy[:, 0], o_xy[:, 1], c='#2ca02c', s=30, alpha=0.6, 
                   marker='o', edgecolors='none', label='Origins (ID)', zorder=5)
        
        # Single fraud
        fraud_xy = _to_xy(test_scores_map[fraud_name])
        if len(fraud_xy) > max_points:
            idx = np.random.default_rng(42).choice(len(fraud_xy), max_points, replace=False)
            fraud_xy = fraud_xy[idx]
        ax.scatter(fraud_xy[:, 0], fraud_xy[:, 1], c='#d62728', s=25, alpha=0.7,
                   marker='x', linewidths=0.8, label=fraud_name, zorder=6)
        
        # GMM centers
        ax.scatter(means_original[:, 0], means_original[:, 1], 
                   c='yellow', s=250, marker='*', edgecolors='black', 
                   linewidths=2, label='GMM Centers', zorder=10)
        
        # Add stats annotation
        ood_p95 = gmm_results['ood_rates'].get(fraud_name, {}).get('p95', 0)
        ood_p99 = gmm_results['ood_rates'].get(fraud_name, {}).get('p99', 0)
        auroc_val = gmm_results['auroc'].get(fraud_name, 0)
        
        stats_text = (f"AUROC: {auroc_val:.3f}\n"
                      f"OOD@p95: {ood_p95:.1%}\n"
                      f"OOD@p99: {ood_p99:.1%}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Reconstruction Cosine Distance', fontsize=11)
        ax.set_ylabel('Sequence Cosine Distance', fontsize=11)
        ax.set_title(f'Origins vs {fraud_name} ({level_name} level)\n'
                     f'GMM Components: {gmm.n_components}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)
        ax.grid(alpha=0.2)
        
        fig.tight_layout()
        if out_dir:
            safe_name = fraud_name.replace(" ", "_")
            fig.savefig(Path(out_dir) / f"gmm_origins_vs_{safe_name}_{level_name}.svg", 
                        format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved gmm_origins_vs_{safe_name}_{level_name}.svg")

    from sklearn.metrics import roc_auc_score

    def _bg(ax):
        ax.contourf(xx, yy, Z, levels=levels, cmap="RdYlGn", alpha=0.65)
        ax.contour(xx, yy, Z, levels=[p95_loglik],
                   colors="darkorange", linewidths=1.8, linestyles="--")
        ax.contour(xx, yy, Z, levels=[p99_loglik],
                   colors="crimson",    linewidths=1.8, linestyles=":")
        ax.scatter(means_original[:, 0], means_original[:, 1],
                   c="gold", s=220, marker="*", edgecolors="black",
                   linewidths=0.8, zorder=12)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Sequence reconstructability\n"
                      "(mean cos. dist. between MSM in/out embeddings)", fontsize=9)
        ax.set_ylabel("Sequence similarity\n"
                      "(mean projector embeddings pairwise cos. dist.)", fontsize=9)
        ax.grid(alpha=0.2, linewidth=0.5, linestyle=":")


    origins_test_xy = _to_xy(test_scores_map["origins"])
    fraud_names = [n for n in test_scores_map if n != "origins"]
    for fraud_name in fraud_names:
        fraud_xy = _to_xy(test_scores_map[fraud_name])

        # Compute AUROC on the fly (negated log-lik as anomaly score)
        s_orig  = -gmm.score_samples(scaler.transform(origins_test_xy))
        s_fraud = -gmm.score_samples(scaler.transform(fraud_xy))
        y_true  = np.concatenate([np.zeros(len(s_orig)), np.ones(len(s_fraud))])
        try:
            auroc = roc_auc_score(y_true, np.concatenate([s_orig, s_fraud]))
        except Exception:
            auroc = float("nan")

        color  = GROUP_COLORS.get(fraud_name, "#d62728")
        marker = GROUP_MARKERS.get(fraud_name, "x")

        fig, (ax_val, ax_test) = plt.subplots(
            1, 2, figsize=(15, 6), gridspec_kw={"wspace": 0.28}
        )

        # ── LEFT: validation set (legits only) ───────────────────────────────
        _bg(ax_val)
        o_val = _subsample(origins_xy, max_points)
        ax_val.scatter(o_val[:, 0], o_val[:, 1],
                       c=GROUP_COLORS["origins"], s=35, alpha=0.75,
                       marker="o", edgecolors="white", linewidths=0.3, zorder=6)
        ax_val.set_title("Validation Set (Legits Only)\nGMM density",
                         fontsize=10, fontweight="bold")
        ax_val.legend(
            handles=[
                mlines.Line2D([], [], marker="o", color=GROUP_COLORS["origins"],
                              markersize=6, linestyle="None", label="Legits (val)"),
                mlines.Line2D([], [], color="darkorange", lw=1.5,
                              linestyle="--", label="p95 boundary"),
                mlines.Line2D([], [], color="crimson",    lw=1.5,
                              linestyle=":",  label="p99 boundary"),
            ],
            fontsize=7.5, loc="lower right", framealpha=0.85,
            title="GMM boundary",
        )

        # ── RIGHT: test set (legits + one fraud) ──────────────────────────────
        _bg(ax_test)
        o_test = _subsample(origins_test_xy, max_points)
        ax_test.scatter(o_test[:, 0], o_test[:, 1],
                        c=GROUP_COLORS["origins"], s=35, alpha=0.75,
                        marker="o", edgecolors="white", linewidths=0.3, zorder=6)
        fr_xy = _subsample(fraud_xy, max_points, seed=1)
        ax_test.scatter(fr_xy[:, 0], fr_xy[:, 1],
                        c=color, s=45, alpha=0.80, marker=marker,
                        linewidths=0.8, zorder=7)
        ax_test.set_title("Test Set (Legits & Attacks)\nLikelihood frontiers",
                          fontsize=10, fontweight="bold")

        # Classes legend (outside, top-right)
        leg1 = ax_test.legend(
            handles=[
                mlines.Line2D([], [], marker="o", color=GROUP_COLORS["origins"],
                              markersize=6, linestyle="None", label="Legits (test)"),
                mlines.Line2D([], [], marker=marker, color=color, markersize=7,
                              linestyle="None",
                              label=f"{fraud_name} (AUROC={auroc:.2f})"),
            ],
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            fontsize=7.5, title="Classes", framealpha=0.88,
        )
        ax_test.add_artist(leg1)
        # Boundaries legend (bottom-right)
        ax_test.legend(
            handles=[
                mlines.Line2D([], [], color="darkorange", lw=1.5,
                              linestyle="--",
                              label=f"OOD@p95 (val thr={p95_loglik:.3f})"),
                mlines.Line2D([], [], color="crimson", lw=1.5,
                              linestyle=":",
                              label=f"OOD@p99 (val thr={p99_loglik:.3f})"),
            ],
            loc="lower right", fontsize=7.5,
            title="Boundaries (val-calibrated)", framealpha=0.85,
        )

        # Suptitle
        fig.suptitle(
            f"GMM OOD Detection  —  {level_name} level  "
            f"|  k={gmm.n_components}",
            fontsize=12, fontweight="bold", y=1.01,
        )

        if out_dir:
            safe = fraud_name.replace(" ", "_")
            fig.savefig(
                Path(out_dir) / f"gmm_sidebyside_{safe}_{level_name}.png",
                dpi=150, bbox_inches="tight",
            )
            fig.savefig(
                Path(out_dir) / f"gmm_sidebyside_{safe}_{level_name}.svg",
                bbox_inches="tight",
            )
            print(f"  ✓ Saved gmm_sidebyside_{safe}_{level_name}.png/svg")
        plt.close(fig)


# viz/gmm_grouped.py
"""GMM plotting at grouped level."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, Any

from eval.grouping import FRAUD_GROUPS


# Color scheme for groups
GROUP_COLORS = {
    "origins": "#2ca02c",              # green
    "static_midvholo": "#1f77b4",      # blue
    "static_midvdynattack": "#ff7f0e", # orange
    "dynamic_midvdynattack": "#d62728",# red
    "swap_midvdynattack": "#9467bd",   # purple
    "photo_replacement": "#8c564b",    # brown
}

GROUP_MARKERS = {
    "static_midvholo":        "x",
    "static_midvdynattack":   "^",
    "dynamic_midvdynattack":  "s",
    "swap_midvdynattack":     "D",
    "photo_replacement":      "P",
}

def plot_gmm_grouped(
    gmm_results: Dict[str, Any],
    grouped_scores: Dict[str, Dict[str, np.ndarray]],
    origins_scores: Dict[str, np.ndarray],
    stats: Dict[str, Any],
    level_name: str = "sequence",
    out_dir: str | Path | None = None,
    max_points: int = 500,
    val_scores={},
):
    """Plot GMM with fraud groups color-coded."""
    
    gmm = gmm_results["gmm"]
    scaler = gmm_results["scaler"]
    thresholds = gmm_results["thresholds"]
    
    def _to_xy(d):
        return np.vstack([d["recon_cosine"], d["seq_cosine"]]).T
    
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # ── Compute grid ──────────────────────────────────────────
    origins_xy = _to_xy(origins_scores)
    origins_val_xy = _to_xy(val_scores)
    all_xy = [origins_xy]
    for sdict in grouped_scores.values():
        all_xy.append(_to_xy(sdict))
    all_xy = np.vstack(all_xy)
    
    margin = 0.05 * (all_xy.max(axis=0) - all_xy.min(axis=0))
    x_min, x_max = all_xy[:, 0].min() - margin[0], all_xy[:, 0].max() + margin[0]
    y_min, y_max = all_xy[:, 1].min() - margin[1], all_xy[:, 1].max() + margin[1]
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    grid_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = gmm.score_samples(grid_scaled).reshape(xx.shape)
    
    p95_loglik = -thresholds["p95"]
    p99_loglik = -thresholds["p99"]
    
    # ══════════════════════════════════════════════════════════
    # PLOT: All groups together
    # ══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Contours
    levels = np.linspace(Z.min(), Z.max(), 25)
    ax.contourf(xx, yy, Z, levels=levels, cmap='Greys', alpha=0.3)
    ax.contour(xx, yy, Z, levels=[p95_loglik], colors='orange', linewidths=2.5)
    ax.contour(xx, yy, Z, levels=[p99_loglik], colors='red', linewidths=2.5, linestyles='--')
    
    # Origins
    o_xy = origins_xy
    if len(o_xy) > max_points:
        idx = np.random.default_rng(42).choice(len(o_xy), max_points, replace=False)
        o_xy = o_xy[idx]
    ax.scatter(o_xy[:, 0], o_xy[:, 1], c=GROUP_COLORS["origins"], s=25, alpha=0.5,
               marker='o', edgecolors='none', label='Origins', zorder=5)
    
    # Each group
    markers = ['s', '^', 'D', 'v', 'P']
    for i, (group_name, _) in enumerate(FRAUD_GROUPS.items()):
        if group_name not in grouped_scores:
            continue
        
        xy = _to_xy(grouped_scores[group_name])
        if len(xy) > max_points:
            idx = np.random.default_rng(42 + i).choice(len(xy), max_points, replace=False)
            xy = xy[idx]
        
        color = GROUP_COLORS.get(group_name, f"C{i}")
        marker = markers[i % len(markers)]
        
        # Get stats for label
        m = stats["grouped"].get(group_name, {})
        auroc_val = m.get("auroc", 0)
        ood95 = m.get("ood_p95", 0)
        n = m.get("n_samples", len(xy))
        
        label = f"{group_name}\n(n={n}, AUROC={auroc_val:.2f}, OOD@95={ood95:.0%})"
        
        ax.scatter(xy[:, 0], xy[:, 1], c=color, s=30, alpha=0.6,
                   marker=marker, label=label, zorder=6)
    
    # GMM centers
    means_orig = scaler.inverse_transform(gmm.means_)
    ax.scatter(means_orig[:, 0], means_orig[:, 1], c='yellow', s=400, marker='*',
               edgecolors='black', linewidths=2, zorder=10, label='GMM Centers')
    
    ax.set_xlabel('Reconstruction Cosine Distance', fontsize=12)
    ax.set_ylabel('Sequence Cosine Distance', fontsize=12)
    ax.set_title(f'GMM Anomaly Detection by Attack Group ({level_name} level)\n'
                 f'Orange = p95 threshold | Red = p99 threshold',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(alpha=0.2)
    
    fig.tight_layout()
    if out_dir:
        fig.savefig(Path(out_dir) / f"gmm_grouped_{level_name}.svg", bbox_inches='tight')
        fig.savefig(Path(out_dir) / f"gmm_grouped_{level_name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved gmm_grouped_{level_name}.svg/png")
    
    # ══════════════════════════════════════════════════════════
    # PLOT: One per group (origins vs group)
    # ══════════════════════════════════════════════════════════
    for group_name in FRAUD_GROUPS.keys():
        if group_name not in grouped_scores:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Contours
        ax.contourf(xx, yy, Z, levels=levels, cmap='viridis', alpha=0.4)
        ax.contour(xx, yy, Z, levels=[p95_loglik], colors='orange', linewidths=2.5)
        ax.contour(xx, yy, Z, levels=[p99_loglik], colors='red', linewidths=2.5, linestyles='--')
        
        # Origins
        ax.scatter(o_xy[:, 0], o_xy[:, 1], c=GROUP_COLORS["origins"], s=30, alpha=0.5,
                   marker='o', edgecolors='none', label='Origins', zorder=5)
        
        # Group
        xy = _to_xy(grouped_scores[group_name])
        if len(xy) > max_points * 2:
            idx = np.random.default_rng(42).choice(len(xy), max_points * 2, replace=False)
            xy = xy[idx]
        
        color = GROUP_COLORS.get(group_name, "red")
        ax.scatter(xy[:, 0], xy[:, 1], c=color, s=30, alpha=0.6,
                   marker='x', linewidths=0.8, label=group_name, zorder=6)
        
        # GMM centers
        ax.scatter(means_orig[:, 0], means_orig[:, 1], c='yellow', s=300, marker='*',
                   edgecolors='black', linewidths=2, zorder=10)
        
        # Stats box
        m = stats["grouped"].get(group_name, {})
        stats_text = (f"N = {m.get('n_samples', 0)}\n"
                      f"AUROC = {m.get('auroc', 0):.3f}\n"
                      f"OOD@p95 = {m.get('ood_p95', 0):.1%}\n"
                      f"OOD@p99 = {m.get('ood_p99', 0):.1%}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_xlabel('Reconstruction Cosine Distance', fontsize=11)
        ax.set_ylabel('Sequence Cosine Distance', fontsize=11)
        ax.set_title(f'Origins vs {group_name} ({level_name} level)', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=10)
        ax.grid(alpha=0.2)
        
        fig.tight_layout()
        if out_dir:
            fig.savefig(Path(out_dir) / f"gmm_origins_vs_{group_name}_{level_name}.svg", bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved gmm_origins_vs_{group_name}_{level_name}.svg")


    # ── Helper: shared background ─────────────────────────────────────────────
    def _bg(ax):
        ax.contourf(xx, yy, Z, levels=levels, cmap="RdYlGn", alpha=0.65)
        ax.contour(xx, yy, Z, levels=[p95_loglik],
                   colors="darkorange", linewidths=1.8, linestyles="--")
        ax.contour(xx, yy, Z, levels=[p99_loglik],
                   colors="crimson",    linewidths=1.8, linestyles=":")
        ax.scatter(means_orig[:, 0], means_orig[:, 1],
                   c="gold", s=220, marker="*", edgecolors="black",
                   linewidths=0.8, zorder=12)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Sequence reconstructability\n"
                      "(mean cos. dist. between MSM in/out embeddings)", fontsize=12)
        ax.set_ylabel("Sequence similarity\n"
                      "(mean projector embeddings pairwise cos. dist.)", fontsize=12)
        ax.grid(alpha=0.2, linewidth=0.5, linestyle=":")

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 2 (NEW): side-by-side — val | test + all groups + AUROCs
    # ════════════════════════════════════════════════════════════════════════
    fig, (ax_val, ax_test) = plt.subplots(
        1, 2, figsize=(16, 6.5), gridspec_kw={"wspace": 0.28}
    )

    # LEFT
    _bg(ax_val)
    ax_val.scatter(
        *_subsample(origins_val_xy, max_points).T,
        c=GROUP_COLORS["origins"], s=35, alpha=0.75,
        marker="o", edgecolors="white", linewidths=0.3, zorder=6,
    )
    ax_val.set_title("Validation Set (Legits Only)\nGMM density",
                     fontsize=10, fontweight="bold")
    ax_val.legend(
        handles=[
            mlines.Line2D([], [], marker="o", color=GROUP_COLORS["origins"],
                          markersize=6, linestyle="None", label="Legits (val)"),
            mlines.Line2D([], [], color="darkorange", lw=1.5,
                          linestyle="--", label="p95 boundary"),
            mlines.Line2D([], [], color="crimson",    lw=1.5,
                          linestyle=":",  label="p99 boundary"),
        ],
        fontsize=7.5, loc="lower right", framealpha=0.85, title="GMM boundary",
    )

    # RIGHT
    _bg(ax_test)
    ax_test.scatter(
        *_subsample(origins_xy, max_points).T,
        c=GROUP_COLORS["origins"], s=35, alpha=0.75,
        marker="o", edgecolors="white", linewidths=0.3, zorder=6,
    )
    class_handles = [
        mlines.Line2D([], [], marker="o", color=GROUP_COLORS["origins"],
                      markersize=6, linestyle="None", label="Legits (test)"),
    ]
    _markers = ["s", "^", "D", "v", "P"]
    for i, (group_name, _) in enumerate(FRAUD_GROUPS.items()):
        if group_name not in grouped_scores:
            continue
        color  = GROUP_COLORS.get(group_name, f"C{i}")
        marker = GROUP_MARKERS.get(group_name, _markers[i % len(_markers)])
        xy     = _to_xy(grouped_scores[group_name])
        s_orig  = -gmm.score_samples(scaler.transform(origins_xy))
        s_fraud = -gmm.score_samples(scaler.transform(xy))
        y_true  = np.concatenate([np.zeros(len(s_orig)), np.ones(len(s_fraud))])
        try:
            auroc = roc_auc_score(y_true, np.concatenate([s_orig, s_fraud]))
        except Exception:
            auroc = float("nan")
        ax_test.scatter(
            *_subsample(xy, max_points, 42 + i).T,
            c=color, s=48, alpha=0.80, marker=marker, linewidths=0.8, zorder=7,
        )
        class_handles.append(
            mlines.Line2D([], [], marker=marker, color=color, markersize=7,
                          linestyle="None",
                          label=f"{group_name} (AUROC={auroc:.2f})")
        )
    ax_test.set_title("Test Set (Legits & Attacks)\
                      ",
                      fontsize=12, fontweight="bold")
    leg1 = ax_test.legend(
        handles=class_handles, loc="lower right",
        bbox_to_anchor=(0.16, 0.8),
        fontsize=7,
        title="Classes", framealpha=0.88,
    )
    ax_test.add_artist(leg1)
    # ax_test.legend(
    #     handles=[
    #         mlines.Line2D([], [], color="darkorange", lw=1.5,
    #                       linestyle="--",
    #                       label=f"OOD@p95 (val thr={p95_loglik:.3f})"),
    #         mlines.Line2D([], [], color="crimson",    lw=1.5,
    #                       linestyle=":",
    #                       label=f"OOD@p99 (val thr={p99_loglik:.3f})"),
    #     ],
    #     loc="lower right", fontsize=7.5,
    #     title="Boundaries (val-calibrated)", framealpha=0.85,
    # )

    fig.suptitle(
        f"GMM OOD Detection  —  {level_name} level  |  k={gmm.n_components}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # fig.tight_layout()
    if out_dir:
        stem = f"gmm_grouped_sidebyside_{level_name}"
        fig.savefig(Path(out_dir) / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(Path(out_dir) / f"{stem}.pdf", dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved {stem}.svg/pdf")
    plt.close(fig)


def plot_auroc_bar_grouped(
    stats: Dict[str, Any],
    level_name: str = "sequence",
    out_dir: str | Path | None = None,
):
    """Bar chart comparing AUROC across groups."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    groups = [g for g in FRAUD_GROUPS.keys() if g in stats["grouped"]]
    aurocs = [stats["grouped"][g]["auroc"] for g in groups]
    colors = [GROUP_COLORS.get(g, "gray") for g in groups]
    n_samples = [stats["grouped"][g]["n_samples"] for g in groups]
    
    bars = ax.bar(range(len(groups)), aurocs, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add sample counts on bars
    for i, (bar, n) in enumerate(zip(bars, n_samples)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'n={n}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(0.9, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title(f'AUROC by Attack Group ({level_name} level)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"auroc_grouped_{level_name}.svg", bbox_inches='tight')
        fig.savefig(Path(out_dir) / f"auroc_grouped_{level_name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved auroc_grouped_{level_name}.svg/png")


"""Side-by-side GMM OOD Detection plot.

Layout (mirrors plot_ocsvm.py):
  LEFT  — Validation set (legit only) + GMM density + p95/p99 contours
  RIGHT — Test set: legit origins (test split) + per-fraud scatter + AUROCs

Signal names expected in score dicts:
  "recon_cosine"  → x-axis  (Sequence reconstructability)
  "seq_cosine"    → y-axis  (Sequence similarity)

Public entry-points
-------------------
plot_gmm_sidebyside_all_frauds(gmm_results, id_scores, test_scores_map, ...)
    One figure per fraud: val panel | test panel (origins + that fraud).

plot_gmm_sidebyside_grouped(gmm_results, id_scores, grouped_scores,
                             origins_test_scores, stats, ...)
    One figure for ALL fraud groups overlaid on the right panel.
"""

# from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Any, Dict

# ──────────────────────────────────────────────────────────────────────────────
# Colour / style constants  (mirrors plot_ocsvm.py)
# ──────────────────────────────────────────────────────────────────────────────

GROUP_COLORS = {
    "origins":                "#2ca02c",
    "static_midvholo":        "#1f77b4",
    "static_midvdynattack":   "#ff7f0e",
    "dynamic_midvdynattack":  "#d62728",
    "swap_midvdynattack":     "#9467bd",
    "photo_replacement":      "#8c564b",
}

GROUP_MARKERS = {
    "static_midvholo":       "x",
    "static_midvdynattack":  "^",
    "dynamic_midvdynattack": "s",
    "swap_midvdynattack":    "D",
    "photo_replacement":     "P",
}

# RdYlGn background — same as OCSVM plot
_CMAP = "RdYlGn"


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_xy(d: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack (recon_cosine, seq_cosine) → (N, 2)."""
    return np.vstack([d["recon_cosine"], d["seq_cosine"]]).T


def _subsample(xy: np.ndarray, max_pts: int, seed: int = 42) -> np.ndarray:
    if len(xy) <= max_pts:
        return xy
    rng = np.random.default_rng(seed)
    return xy[rng.choice(len(xy), max_pts, replace=False)]


def _make_grid(
    gmm_results: Dict[str, Any],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate GMM log-likelihood on a 2-D grid (in original space)."""
    gmm    = gmm_results["gmm"]
    scaler = gmm_results["scaler"]
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    xx, yy = np.meshgrid(xs, ys)
    pts_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = gmm.score_samples(pts_scaled).reshape(xx.shape)
    return xx, yy, Z


def _plot_limits(
    *score_dicts: Dict[str, np.ndarray],
    pad: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute shared (x_range, y_range) with a small margin."""
    all_xy = np.vstack([_to_xy(d) for d in score_dicts])
    margin = pad * (all_xy.max(0) - all_xy.min(0))
    x_rng  = (all_xy[:, 0].min() - margin[0], all_xy[:, 0].max() + margin[0])
    y_rng  = (all_xy[:, 1].min() - margin[1], all_xy[:, 1].max() + margin[1])
    return x_rng, y_rng


def _draw_background(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    Z: np.ndarray,
    n_levels: int = 30,
) -> None:
    """Continuous RdYlGn density fill."""
    levels = np.linspace(Z.min(), Z.max(), n_levels)
    ax.contourf(xx, yy, Z, levels=levels, cmap=_CMAP, alpha=0.65)


def _draw_threshold_contours(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    Z: np.ndarray,
    thresholds: Dict[str, float],
    label_contours: bool = True,
) -> list[mlines.Line2D]:
    """
    Draw val-calibrated p95 (dashed orange) and p99 (dotted red) contours.
    Returns legend handles.
    """
    p95_ll = -thresholds["p95"]
    p99_ll = -thresholds["p99"]
    handles = []

    cs95 = ax.contour(xx, yy, Z, levels=[p95_ll],
                      colors="darkorange", linewidths=1.8, linestyles="--")
    cs99 = ax.contour(xx, yy, Z, levels=[p99_ll],
                      colors="crimson",    linewidths=1.8, linestyles=":")

    if label_contours:
        h95 = mlines.Line2D([], [], color="darkorange", lw=1.8,
                            linestyle="--",
                            label=f"OOD@p95 (val thr={p95_ll:.3f})")
        h99 = mlines.Line2D([], [], color="crimson", lw=1.8,
                            linestyle=":",
                            label=f"OOD@p99 (val thr={p99_ll:.3f})")
        handles = [h95, h99]
    return handles


def _draw_gmm_centers(
    ax: plt.Axes,
    gmm_results: Dict[str, Any],
    label: bool = True,
) -> list:
    """Plot GMM component means (in original space)."""
    gmm    = gmm_results["gmm"]
    scaler = gmm_results["scaler"]
    means  = scaler.inverse_transform(gmm.means_)
    h = ax.scatter(
        means[:, 0], means[:, 1],
        c="gold", s=220, marker="*",
        edgecolors="black", linewidths=0.8,
        zorder=12,
        label=f"GMM centers (k={gmm.n_components})" if label else None,
    )
    return [h] if label else []


def _axis_labels(
    ax: plt.Axes,
    title: str,
    xlabel: str = "Sequence reconstructability\n(mean cos. dist. between MSM in/out embeddings)",
    ylabel: str = "Sequence similarity\n(mean projector embeddings pairwise cos. dist.)",
) -> None:
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.2, linewidth=0.5, linestyle=":")

from sklearn.metrics import roc_auc_score

def _auroc_score(
    gmm_results: Dict[str, Any],
    origins_scores: Dict[str, np.ndarray],
    fraud_scores: Dict[str, np.ndarray],
) -> float:
    """Compute AUROC using the GMM anomaly score (negated log-likelihood)."""
    gmm    = gmm_results["gmm"]
    scaler = gmm_results["scaler"]

    X_orig  = scaler.transform(_to_xy(origins_scores))
    X_fraud = scaler.transform(_to_xy(fraud_scores))

    s_orig  = -gmm.score_samples(X_orig)   # high = more OOD
    s_fraud = -gmm.score_samples(X_fraud)

    y_true  = np.concatenate([np.zeros(len(s_orig)), np.ones(len(s_fraud))])
    y_score = np.concatenate([s_orig, s_fraud])
    
    return float(roc_auc_score(y_true, y_score))


# ──────────────────────────────────────────────────────────────────────────────
# Left panel: validation set only
# ──────────────────────────────────────────────────────────────────────────────

def _draw_val_panel(
    ax: plt.Axes,
    gmm_results: Dict[str, Any],
    id_scores: Dict[str, np.ndarray],
    xx: np.ndarray,
    yy: np.ndarray,
    Z: np.ndarray,
    max_points: int = 500,
) -> None:
    """Left panel: GMM density + val legits only."""
    _draw_background(ax, xx, yy, Z)
    _draw_threshold_contours(ax, xx, yy, Z, gmm_results["thresholds"],
                             label_contours=False)
    _draw_gmm_centers(ax, gmm_results, label=False)

    val_xy = _subsample(_to_xy(id_scores), max_points)
    ax.scatter(
        val_xy[:, 0], val_xy[:, 1],
        c=GROUP_COLORS["origins"], s=35, alpha=0.75,
        marker="o", edgecolors="white", linewidths=0.3,
        zorder=6, label="Legits (val)",
    )

    # Legend: single item + threshold description in title
    ax.legend(handles=[
        mlines.Line2D([], [], marker="o", color=GROUP_COLORS["origins"],
                      markersize=6, linestyle="None", label="Legits (val)"),
        mlines.Line2D([], [], color="darkorange", lw=1.5,
                      linestyle="--", label="p95 boundary"),
        mlines.Line2D([], [], color="crimson", lw=1.5,
                      linestyle=":",  label="p99 boundary"),
    ], fontsize=7.5, loc="lower right", framealpha=0.85, title="GMM boundary")


# ──────────────────────────────────────────────────────────────────────────────
# Right panel helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_test_panel_single_fraud(
    ax: plt.Axes,
    gmm_results: Dict[str, Any],
    origins_test: Dict[str, np.ndarray],
    fraud_scores: Dict[str, np.ndarray],
    fraud_name: str,
    xx: np.ndarray,
    yy: np.ndarray,
    Z: np.ndarray,
    max_points: int = 500,
) -> None:
    """Right panel: origins (test) + one fraud type + AUROC."""
    _draw_background(ax, xx, yy, Z)
    thr_handles = _draw_threshold_contours(
        ax, xx, yy, Z, gmm_results["thresholds"], label_contours=True)
    _draw_gmm_centers(ax, gmm_results, label=False)

    # Legit test origins
    orig_xy = _subsample(_to_xy(origins_test), max_points)
    h_orig = ax.scatter(
        orig_xy[:, 0], orig_xy[:, 1],
        c=GROUP_COLORS["origins"], s=35, alpha=0.75,
        marker="o", edgecolors="white", linewidths=0.3,
        zorder=6, label="Legits (test)",
    )

    # Fraud scatter
    color  = GROUP_COLORS.get(fraud_name, "#e74c3c")
    marker = GROUP_MARKERS.get(fraud_name, "x")
    fr_xy  = _subsample(_to_xy(fraud_scores), max_points, seed=1)
    auroc  = _auroc_score(gmm_results, origins_test, fraud_scores)
    h_fr   = ax.scatter(
        fr_xy[:, 0], fr_xy[:, 1],
        c=color, s=45, alpha=0.80, marker=marker, linewidths=0.8,
        zorder=7, label=f"{fraud_name} (AUROC={auroc:.2f})",
    )

    # Class legend (top-right, outside)
    class_handles = [
        mlines.Line2D([], [], marker="o", color=GROUP_COLORS["origins"],
                      markersize=6, linestyle="None", label="Legits (test)"),
        mlines.Line2D([], [], marker=marker, color=color,
                      markersize=7, linestyle="None",
                      label=f"{fraud_name} (AUROC={auroc:.2f})"),
    ]
    leg1 = ax.legend(handles=class_handles, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), fontsize=7.5,
                     title="Classes", framealpha=0.88)
    ax.add_artist(leg1)

    # Boundary legend (bottom-right)
    ax.legend(handles=thr_handles, loc="lower right", fontsize=7.5,
              title="Boundaries (val-calibrated)", framealpha=0.85)


def _draw_test_panel_grouped(
    ax: plt.Axes,
    gmm_results: Dict[str, Any],
    origins_test: Dict[str, np.ndarray],
    grouped_scores: Dict[str, Dict[str, np.ndarray]],
    stats: Dict[str, Any],
    xx: np.ndarray,
    yy: np.ndarray,
    Z: np.ndarray,
    max_points: int = 500,
) -> None:
    """Right panel: origins (test) + all fraud groups overlaid + AUROCs."""
    _draw_background(ax, xx, yy, Z)
    thr_handles = _draw_threshold_contours(
        ax, xx, yy, Z, gmm_results["thresholds"], label_contours=True)
    _draw_gmm_centers(ax, gmm_results, label=False)

    # Legit test origins
    orig_xy = _subsample(_to_xy(origins_test), max_points)
    ax.scatter(
        orig_xy[:, 0], orig_xy[:, 1],
        c=GROUP_COLORS["origins"], s=35, alpha=0.75,
        marker="o", edgecolors="white", linewidths=0.3, zorder=6,
    )

    class_handles = [
        mlines.Line2D([], [], marker="o", color=GROUP_COLORS["origins"],
                      markersize=6, linestyle="None", label="Legits (test)"),
    ]

    markers_cycle = ["x", "^", "s", "D", "P", "v"]
    for i, (gname, gscores) in enumerate(grouped_scores.items()):
        color  = GROUP_COLORS.get(gname, f"C{i}")
        marker = GROUP_MARKERS.get(gname, markers_cycle[i % len(markers_cycle)])
        fr_xy  = _subsample(_to_xy(gscores), max_points, seed=42 + i)
        auroc  = _auroc_score(gmm_results, origins_test, gscores)

        ax.scatter(
            fr_xy[:, 0], fr_xy[:, 1],
            c=color, s=48, alpha=0.80, marker=marker, linewidths=0.8,
            zorder=7,
        )
        class_handles.append(
            mlines.Line2D([], [], marker=marker, color=color, markersize=7,
                          linestyle="None",
                          label=f"{gname} (AUROC={auroc:.2f})")
        )

    leg1 = ax.legend(handles=class_handles, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), fontsize=7,
                     title="Classes", framealpha=0.88)
    ax.add_artist(leg1)
    ax.legend(handles=thr_handles, loc="lower right", fontsize=7.5,
              title="Boundaries (val-calibrated)", framealpha=0.85)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def plot_gmm_sidebyside_all_frauds(
    gmm_results: Dict[str, Any],
    id_scores: Dict[str, np.ndarray],
    test_scores_map: Dict[str, Dict[str, np.ndarray]],
    level_name: str = "sequence",
    out_dir: str | Path | None = None,
    max_points: int = 500,
    nu_label: str | None = None,
) -> None:
    """
    One figure per fraud: LEFT = val legits, RIGHT = test legits + that fraud.

    Parameters
    ----------
    gmm_results     : output of fit_gmm_and_score()
    id_scores       : validation legit scores  {signal -> array}
    test_scores_map : {name -> {signal -> array}} — must include key 'origins'
                      which holds the test-split legit scores.
    level_name      : 'sequence' or 'video'
    out_dir         : directory to save .svg/.png (optional)
    max_points      : max scatter points per class
    nu_label        : extra string for suptitle (e.g. GMM component count)
    """
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    origins_test  = test_scores_map.get("origins", id_scores)
    fraud_names   = [n for n in test_scores_map if n != "origins"]
    gmm           = gmm_results["gmm"]

    # Shared axis limits & grid (computed once)
    all_dicts = [id_scores, origins_test] + [
        test_scores_map[n] for n in fraud_names]
    x_rng, y_rng = _plot_limits(*all_dicts)
    xx, yy, Z    = _make_grid(gmm_results, x_rng, y_rng)

    for fraud_name in fraud_names:
        fig, axes = plt.subplots(
            1, 2,
            figsize=(15, 6),
            gridspec_kw={"wspace": 0.30},
        )
        ax_val, ax_test = axes

        # ── Left panel ────────────────────────────────────────
        _draw_val_panel(ax_val, gmm_results, id_scores, xx, yy, Z, max_points)
        ax_val.set_xlim(*x_rng); ax_val.set_ylim(*y_rng)
        _axis_labels(
            ax_val,
            title="Validation Set (Legits Only)\nGMM density",
        )

        # ── Right panel ───────────────────────────────────────
        _draw_test_panel_single_fraud(
            ax_test, gmm_results, origins_test,
            test_scores_map[fraud_name], fraud_name,
            xx, yy, Z, max_points,
        )
        ax_test.set_xlim(*x_rng); ax_test.set_ylim(*y_rng)
        _axis_labels(
            ax_test,
            title="Test Set (Legits & Attacks)\nLikelihood frontiers",
        )

        # ── Suptitle ──────────────────────────────────────────
        k_str = f"k={gmm.n_components}" + (f" | {nu_label}" if nu_label else "")
        fig.suptitle(
            f"GMM OOD Detection  —  {level_name} level  |  {k_str}",
            fontsize=12, fontweight="bold", y=1.01,
        )

        if out_dir:
            safe = fraud_name.replace(" ", "_")
            fig.savefig(
                Path(out_dir) / f"gmm_sidebyside_{safe}_{level_name}.png",
                dpi=150, bbox_inches="tight",
            )
            fig.savefig(
                Path(out_dir) / f"gmm_sidebyside_{safe}_{level_name}.svg",
                bbox_inches="tight",
            )
            print(f"  ✓ Saved gmm_sidebyside_{safe}_{level_name}.png/svg")
        plt.close(fig)


def plot_gmm_sidebyside_grouped(
    gmm_results: Dict[str, Any],
    id_scores: Dict[str, np.ndarray],
    grouped_scores: Dict[str, Dict[str, np.ndarray]],
    origins_test_scores: Dict[str, np.ndarray],
    stats: Dict[str, Any],
    level_name: str = "sequence",
    out_dir: str | Path | None = None,
    max_points: int = 500,
    nu_label: str | None = None,
) -> None:
    """
    Single figure: LEFT = val legits, RIGHT = test legits + all fraud groups.

    Parameters
    ----------
    gmm_results         : output of fit_gmm_and_score()
    id_scores           : validation legit scores {signal -> array}
    grouped_scores      : {group_name -> {signal -> array}}  (fraud groups)
    origins_test_scores : test-split legit scores {signal -> array}
    stats               : output of compute_stats_all_levels() (optional,
                          used only for tooltip strings if extended)
    level_name          : 'sequence' or 'video'
    out_dir             : save directory (optional)
    max_points          : max scatter points per class
    nu_label            : extra suptitle annotation
    """
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    gmm = gmm_results["gmm"]

    all_dicts = [id_scores, origins_test_scores] + list(grouped_scores.values())
    x_rng, y_rng = _plot_limits(*all_dicts)
    xx, yy, Z    = _make_grid(gmm_results, x_rng, y_rng)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 6.5),
        gridspec_kw={"wspace": 0.30},
    )
    ax_val, ax_test = axes

    # ── Left panel ────────────────────────────────────────────
    _draw_val_panel(ax_val, gmm_results, id_scores, xx, yy, Z, max_points)
    ax_val.set_xlim(*x_rng); ax_val.set_ylim(*y_rng)
    _axis_labels(ax_val, title="Validation Set (Legits Only)\nGMM density")

    # ── Right panel ───────────────────────────────────────────
    _draw_test_panel_grouped(
        ax_test, gmm_results, origins_test_scores,
        grouped_scores, stats, xx, yy, Z, max_points,
    )
    ax_test.set_xlim(*x_rng); ax_test.set_ylim(*y_rng)
    _axis_labels(ax_test,
                 title="Test Set (Legits & Attacks)\nLikelihood frontiers")

    # ── Suptitle ──────────────────────────────────────────────
    k_str = f"k={gmm.n_components}" + (f" | {nu_label}" if nu_label else "")
    fig.suptitle(
        f"GMM OOD Detection  —  {level_name} level  |  {k_str}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    if out_dir:
        fname = f"gmm_sidebyside_grouped_{level_name}"
        fig.savefig(
            Path(out_dir) / f"{fname}.png",
            dpi=150, bbox_inches="tight",
        )
        fig.savefig(
            Path(out_dir) / f"{fname}.svg",
            bbox_inches="tight",
        )
        print(f"  ✓ Saved {fname}.png/svg")
    plt.close(fig)
