from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def _build_palette(names: list[str]) -> list[str]:
    """First entry (origins) = green, rest = sequential warm colours."""
    n = len(names)
    cmap = plt.cm.get_cmap("tab20", max(n, 2))
    colours = []
    for i in range(n):
        if i == 0:
            colours.append("#2ca02c")  # green for genuine
        else:
            colours.append(cmap((i - 1) / max(n - 1, 1)))
    return colours


def _short_name(name: str, max_len: int = 18) -> str:
    """Truncate long fraud names for axis labels."""
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def plot_violin(
    all_scores: dict[str, dict[str, np.ndarray]],
    signal_key: str,
    title: str,
    ylabel: str,
    out_path: str,
    ref_thresholds: dict[str, dict] | None = None,
):
    """Violin + box plot for one signal across all fraud types."""
    names = [n for n in all_scores if signal_key in all_scores[n]]
    if not names:
        print(f"  ⚠  No data for signal '{signal_key}' — skipping plot.")
        return

    data = [all_scores[n][signal_key] for n in names]
    short_names = [_short_name(n) for n in names]
    colours = _build_palette(names)

    fig, ax = plt.subplots(figsize=(max(len(names) * 0.9, 8), 6))

    parts = ax.violinplot(
        data,
        positions=range(len(names)),
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colours[i])
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.35)

    bp = ax.boxplot(
        data,
        positions=range(len(names)),
        widths=0.2,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.4),
        medianprops=dict(color="black", linewidth=1.2),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colours[i])
        patch.set_alpha(0.7)

    if ref_thresholds and signal_key in ref_thresholds:
        th = ref_thresholds[signal_key]
        p95 = th["percentiles"]["p95"]
        p99 = th["percentiles"]["p99"]
        ax.axhline(p95, color="orange", linestyle="--", linewidth=1, label=f"val p95 = {p95:.5f}")
        ax.axhline(p99, color="red", linestyle="--", linewidth=1, label=f"val p99 = {p99:.5f}")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out_path}")

def plot_cosine_reco_vs_cosine_seq(
    all_scores: dict[str, dict[str, np.ndarray]],
    out_path: str,
    ref_thresholds: dict[str, dict] | None = None,
    max_points_per_class: int = 500,
):
    """2-D scatter: cosine distance (x) vs reconstruction MSE (y)."""
    names = list(all_scores.keys())
    colours = _build_palette(names)

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, name in enumerate(names):
        cos = all_scores[name]["recon_cosine"]
        mse = all_scores[name]["seq_cosine"]

        n = len(cos)
        if n > max_points_per_class:
            idx = np.random.default_rng(42).choice(n, max_points_per_class, replace=False)
            cos, mse = cos[idx], mse[idx]

        marker = "o" if i == 0 else "x"
        size = 30 if i == 0 else 14
        alpha = 0.7 if i == 0 else 0.45
        zorder = 10 if i == 0 else 5

        ax.scatter(
            cos, mse,
            c=[colours[i]],
            marker=marker,
            s=size,
            alpha=alpha,
            linewidths=0.4,
            edgecolors="none" if i == 0 else "face",
            label=_short_name(name),
            zorder=zorder,
        )

    if ref_thresholds:
        cos_p95 = ref_thresholds.get("recon_cosine", {}).get("percentiles", {}).get("p95")
        mse_p95 = ref_thresholds.get("seq_cosine", {}).get("percentiles", {}).get("p95")
        if cos_p95 is not None and mse_p95 is not None:
            ax.axvline(cos_p95, color="orange", ls="--", lw=0.8, alpha=0.7, label=f"cos p95={cos_p95:.4f}")
            ax.axhline(mse_p95, color="orange", ls="--", lw=0.8, alpha=0.7, label=f"mse p95={mse_p95:.4f}")

        cos_p99 = ref_thresholds.get("recon_cosine", {}).get("percentiles", {}).get("p99")
        mse_p99 = ref_thresholds.get("seq_cosine", {}).get("percentiles", {}).get("p99")
        if cos_p99 is not None and mse_p99 is not None:
            ax.axvline(cos_p99, color="red", ls="--", lw=0.8, alpha=0.7, label=f"cos p99={cos_p99:.4f}")
            ax.axhline(mse_p99, color="red", ls="--", lw=0.8, alpha=0.7, label=f"mse p99={mse_p99:.4f}")

    ax.set_xlabel("Cosine Distance reconstruction", fontsize=11)
    ax.set_ylabel("Sequence cosine distance", fontsize=11)
    ax.set_title("Reconstruction Cosine Distance vs Sequence cosine distance", fontsize=13, fontweight="bold")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        framealpha=0.9,
        borderaxespad=0,
    )
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", scilimits=(-2, 3))

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out_path}")


def plot_mse_reco_vs_seq(
    all_scores: dict[str, dict[str, np.ndarray]],
    out_path: str,
    ref_thresholds: dict[str, dict] | None = None,
):
    """Hex-bin density version — useful when scatter is too crowded."""
    names = list(all_scores.keys())
    colours = _build_palette(names)

    fig, ax = plt.subplots(figsize=(10, 7))

    if "seq_mse" in all_scores[names[0]]:
        cos0 = all_scores[names[0]]["seq_mse"]
        mse0 = all_scores[names[0]]["recon_mse"]
        hb = ax.hexbin(cos0, mse0, gridsize=40, cmap="Greens", mincnt=1, alpha=0.6, zorder=1)
        cb = fig.colorbar(hb, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label("origins count", fontsize=8)

    for i, name in enumerate(names[1:], start=1):
        cos = all_scores[name]["seq_mse"]
        mse = all_scores[name]["recon_mse"]
        n = len(cos)
        if n > 300:
            idx = np.random.default_rng(42).choice(n, 300, replace=False)
            cos, mse = cos[idx], mse[idx]
        ax.scatter(cos, mse, c=[colours[i]], marker="x", s=12, alpha=0.5,
                   label=_short_name(name), zorder=5, linewidths=0.5)

    if ref_thresholds:
        for pkey, colour in [("p95", "orange"), ("p99", "red")]:
            cp = ref_thresholds.get("seq_mse", {}).get("percentiles", {}).get(pkey)
            mp = ref_thresholds.get("recon_mse", {}).get("percentiles", {}).get(pkey)
            if cp is not None:
                ax.axvline(cp, color=colour, ls="--", lw=0.8, alpha=0.7)
            if mp is not None:
                ax.axhline(mp, color=colour, ls="--", lw=0.8, alpha=0.7)

    ax.set_xlabel("Sequence MSE", fontsize=11)
    ax.set_ylabel("Reconstruction MSE", fontsize=11)
    ax.set_title("Cosine vs MSE — Density (origins) + Scatter (frauds)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=7,
              framealpha=0.9, borderaxespad=0)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out_path}")


def plot_auroc_bar(
    all_results: dict[str, dict],
    out_path: str,
    title: str = "AUROC per Signal per Fraud Type",
):
    """Grouped bar chart: AUROC per signal per fraud type (skip origins)."""
    fraud_names = [n for n in all_results if n != "origins"]
    if not fraud_names:
        return

    signal_keys = list(all_results[fraud_names[0]]["signals"].keys())

    fig, ax = plt.subplots(figsize=(max(len(fraud_names) * 0.9, 8), 5))
    x = np.arange(len(fraud_names))
    width = 0.8 / len(signal_keys)
    cmap = plt.cm.get_cmap("Set2", len(signal_keys))

    for j, sig in enumerate(signal_keys):
        vals = [all_results[n]["signals"][sig]["auroc"] for n in fraud_names]
        offset = (j - len(signal_keys) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=sig, color=cmap(j), edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(n) for n in fraud_names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUROC", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", ls=":", lw=0.8, alpha=0.5)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out_path}")


# ──────────────────────────────────────────────────────────────
# Per-video plots
# ──────────────────────────────────────────────────────────────

def plot_video_strip(
    all_video_scores: dict[str, dict[str, np.ndarray]],
    all_video_ids: dict[str, list[str]],
    signal_key: str,
    title: str,
    ylabel: str,
    out_path: str,
    ref_thresholds: dict[str, dict] | None = None,
):
    """Strip plot: each dot = one video's mean score.  Useful for spotting outliers."""
    names = [n for n in all_video_scores if signal_key in all_video_scores[n]]
    if not names:
        print(f"  ⚠  No data for signal '{signal_key}' — skipping strip plot.")
        return

    colours = _build_palette(names)
    short_names = [_short_name(n) for n in names]

    fig, ax = plt.subplots(figsize=(max(len(names) * 0.9, 8), 6))

    for i, name in enumerate(names):
        vals = all_video_scores[name][signal_key]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            c=[colours[i]],
            s=28,
            alpha=0.65,
            edgecolors="black",
            linewidths=0.3,
            zorder=5,
        )
        ax.hlines(
            vals.mean(), i - 0.25, i + 0.25,
            colors="black", linewidths=1.5, zorder=6,
        )

    if ref_thresholds and signal_key in ref_thresholds:
        th = ref_thresholds[signal_key]
        p95 = th["percentiles"]["p95"]
        p99 = th["percentiles"]["p99"]
        ax.axhline(p95, color="orange", ls="--", lw=1, label=f"val p95 = {p95:.5f}")
        ax.axhline(p99, color="red", ls="--", lw=1, label=f"val p99 = {p99:.5f}")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {out_path}")




def plot_origins_vs_fraud(
    all_scores: dict[str, dict[str, np.ndarray]],
    out_dir: str,
    ref_thresholds: dict[str, dict] | None = None,
    max_points_per_class: int = 500,
    level: str = "sequence",
):
    """Pairwise scatter: origins vs each fraud type.

    For every fraud type, produce one plot with:
      x = cosine mean (recon_cosine)
      y = MSE mean   (recon_mse)
    Origins are shown in green, the fraud type in red.

    Plots are saved into *out_dir* (one SVG per fraud type).

    Parameters
    ----------
    all_scores : mapping  name → {signal → 1-D array}
    out_dir    : directory in which to save the SVG files
    ref_thresholds : optional validation thresholds for p95/p99 lines
    max_points_per_class : subsample cap per class
    level : "sequence" or "video" — used only in titles
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if "origins" not in all_scores:
        print("  ⚠  'origins' not found in scores — skipping pairwise plots.")
        return

    origins_cos = all_scores["origins"]["recon_cosine"]
    origins_mse = all_scores["origins"]["seq_cosine"]

    fraud_names = [n for n in all_scores if n != "origins"]
    if not fraud_names:
        return

    for fraud_name in fraud_names:
        fraud_cos = all_scores[fraud_name]["recon_cosine"]
        fraud_mse = all_scores[fraud_name]["seq_cosine"]

        fig, ax = plt.subplots(figsize=(8, 6))

        # ── subsample if needed ──────────────────────────────
        def _maybe_subsample(cos, mse, cap):
            n = len(cos)
            if n > cap:
                idx = np.random.default_rng(42).choice(n, cap, replace=False)
                return cos[idx], mse[idx]
            return cos, mse

        o_cos, o_mse = _maybe_subsample(origins_cos, origins_mse, max_points_per_class)
        f_cos, f_mse = _maybe_subsample(fraud_cos, fraud_mse, max_points_per_class)

        # ── scatter ──────────────────────────────────────────
        ax.scatter(
            o_cos, o_mse,
            c="#2ca02c", marker="o", s=28, alpha=0.6,
            edgecolors="none", label="origins", zorder=5,
        )
        ax.scatter(
            f_cos, f_mse,
            c="#d62728", marker="x", s=22, alpha=0.55,
            linewidths=0.6, label=_short_name(fraud_name), zorder=6,
        )

        # ── threshold lines ──────────────────────────────────
        if ref_thresholds:
            cos_p95 = ref_thresholds.get("recon_cosine", {}).get("percentiles", {}).get("p95")
            mse_p95 = ref_thresholds.get("seq_cosine", {}).get("percentiles", {}).get("p95")
            if cos_p95 is not None:
                ax.axvline(cos_p95, color="orange", ls="--", lw=0.8, alpha=0.7,
                           label=f"cos p95={cos_p95:.4f}")
            if mse_p95 is not None:
                ax.axhline(mse_p95, color="orange", ls="--", lw=0.8, alpha=0.7,
                           label=f"mse p95={mse_p95:.4f}")

            cos_p99 = ref_thresholds.get("recon_cosine", {}).get("percentiles", {}).get("p99")
            mse_p99 = ref_thresholds.get("seq_cosine", {}).get("percentiles", {}).get("p99")
            if cos_p99 is not None:
                ax.axvline(cos_p99, color="red", ls="--", lw=0.8, alpha=0.7,
                           label=f"cos p99={cos_p99:.4f}")
            if mse_p99 is not None:
                ax.axhline(mse_p99, color="red", ls="--", lw=0.8, alpha=0.7,
                           label=f"mse p99={mse_p99:.4f}")

        # ── labels / style ───────────────────────────────────
        ax.set_xlabel("Cosine Distance (1 − cos sim) reconstruction", fontsize=11)
        ax.set_ylabel("seq cosine distance", fontsize=11)
        ax.set_title(
            f"Origins vs {_short_name(fraud_name, 30)}  ({level} level)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=8, framealpha=0.9, borderaxespad=0)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", scilimits=(-2, 3))

        fig.tight_layout()
        safe_name = fraud_name.replace(" ", "_")
        out_path = str(Path(out_dir) / f"origins_vs_{safe_name}.svg")
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved {out_path}")
