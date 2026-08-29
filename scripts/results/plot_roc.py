"""plot_roc_comparison.py

Loads ROC data from three different method schemas and produces
the 2x4 multi-panel figure (two main ROC panels + four zoom panels).

Supported method schemas
------------------------
  WSL + MSM   : grouped_stats_video.json
                manifest["grouped"][ds_name]["roc_curve"]["fpr/tpr"]  +  ["auroc"]

  HoloVerif   : roc_curves_grouped.json
                manifest["groups"][group_name]["fpr/tpr/auc"]  (flat arrays, 1 run)

  GMM (folds) : fold_N_roc_curve.json  (canonical schema from fold_roc_save.py)
                manifest["groups"][group]["fpr/tpr/thresholds/auc"]  (lists-of-runs)

Internal unified representation
--------------------------------
  roc_curves : dict
    method_label -> group_key -> {
        "fpr": [np.array, ...],   # one array per run
        "tpr": [np.array, ...],
        "auc": [float, ...],
    }
  where group_key in {"midvholo", "midvdyn", "photorep"}
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Panel metadata
# ─────────────────────────────────────────────────────────────────────────────

PANEL_TITLES = {
    "midvholo": "MIDV-Holo Vanilla (Static Frauds)",
    "midvdyn": "MIDV-DynAttack (Static, Swap & Dynamic Frauds)",
}

ZOOM_TITLES = {
    ("midvholo", "low_fpr"): "MIDV-Holo (FPR ≤ 0.05)",
    ("midvholo", "high_tpr"): "MIDV-Holo (Recall ≥ 0.95)",
    ("midvdyn", "low_fpr"): "MIDV-DynAttack (FPR ≤ 0.05)",
    ("midvdyn", "high_tpr"): "MIDV-DynAttack (Recall ≥ 0.95)",
}

COMMON_FPR = np.linspace(0, 1, 100)


# ─────────────────────────────────────────────────────────────────────────────
# Default group maps  (raw name in JSON  →  canonical group key)
# ─────────────────────────────────────────────────────────────────────────────

WSL_GROUP_MAP: Dict[str, str] = {
    "static_midvholo": "midvholo",
    "midvdynattack": "midvdyn",
    # "static_midvdynattack":  "midvdyn",
    # "dynamic_midvdynattack": "midvdyn",
    # "swap_midvdynattack":    "midvdyn",
    # "photo_replacement":     "photorep",
}

HOLOVERIF_GROUP_MAP: Dict[str, str] = {
    "MIDV_Holo": "midvholo",
    "MIDV-DynAttack": "midvdyn",
}

GMM_GROUP_MAP: Dict[str, str] = {
    "midvholo": "midvholo",
    "midvdyn": "midvdyn",
    "photorep": "photorep",
}


# ─────────────────────────────────────────────────────────────────────────────
# Loader helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_paths(paths) -> list:
    """Accept str, Path, list, or {label: path} dict; always return a list."""
    if isinstance(paths, dict):
        return list(paths.values())
    if isinstance(paths, (str, Path)):
        return [paths]
    return list(paths)


def _add_run(bucket, group, fpr, tpr, auc):
    """Append one run into a group bucket (mutates in-place)."""
    entry = bucket.setdefault(group, {"fpr": [], "tpr": [], "auc": []})
    entry["fpr"].append(np.asarray(fpr, dtype=float))
    entry["tpr"].append(np.asarray(tpr, dtype=float))
    entry["auc"].append(float(auc))


def load_wsl_msm(paths, group_map=None):
    """Load WSL+MSM grouped_stats_video.json — one file per run."""
    if group_map is None:
        group_map = WSL_GROUP_MAP
    result = {}
    for path in _resolve_paths(paths):
        with open(path) as f:
            manifest = json.load(f)
        for ds_name, ds_data in manifest.get("grouped", {}).items():
            group = group_map.get(ds_name)
            if group is None:
                continue
            roc = ds_data.get("roc_curve", {})
            if not roc:
                continue
            fpr = np.array(roc["fpr"])
            tpr = np.array(roc["tpr"])
            # auc = float(ds_data.get("auroc", np.trapz(tpr, fpr)))
            auc = ds_data["auroc"]
            _add_run(result, group, fpr, tpr, auc)
    return result


def load_holoverif(paths, group_map=None):
    """Load HoloVerif roc_curves_grouped.json — one file per run."""
    if group_map is None:
        group_map = HOLOVERIF_GROUP_MAP
    result = {}
    for path in _resolve_paths(paths):
        with open(path) as f:
            manifest = json.load(f)
        for raw_name, g_data in manifest.get("groups", {}).items():
            group = group_map.get(raw_name)
            if group is None:
                continue
            fpr = np.array(g_data["fpr"])
            tpr = np.array(g_data["tpr"])
            # auc = float(g_data.get("auc", np.trapz(tpr, fpr)))
            auc = g_data["auc"]
            _add_run(result, group, fpr, tpr, auc)
    return result


def load_gmm_folds(paths, group_map=None):
    """
    Load fold_N_roc_curve.json files — one file per fold.

    Each fold file may contain multiple dataset-runs per group
    (e.g. 3 midvdyn variants). These are averaged within the fold
    onto COMMON_FPR before being stored as a single run, so that
    5 fold files produce exactly 5 runs per group -> mean ± std band.
    """
    if group_map is None:
        group_map = GMM_GROUP_MAP
    result = {}
    for path in _resolve_paths(paths):
        with open(path) as f:
            manifest = json.load(f)
        for raw_name, g_data in manifest.get("groups", {}).items():
            group = group_map.get(raw_name, raw_name)
            fpr_runs = [np.array(r) for r in g_data["fpr"]]
            tpr_runs = [np.array(r) for r in g_data["tpr"]]
            auc_runs = [a for a in g_data["auc"]]
            # Average dataset-runs within fold onto common grid.
            interp_tprs = [
                np.interp(COMMON_FPR, fpr_r, tpr_r)
                for fpr_r, tpr_r in zip(fpr_runs, tpr_runs)
            ]
            _add_run(
                result,
                group,
                COMMON_FPR,
                np.mean(interp_tprs, axis=0),
                np.mean(auc_runs),
            )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation + mean/std across runs
# ─────────────────────────────────────────────────────────────────────────────


def _mean_std_tpr(fpr_list, tpr_list):
    """
    Interpolate all runs onto COMMON_FPR with np.interp,
    then return (mean_tpr, std_tpr).  std_tpr is zero for a single run.
    """
    interp = np.array(
        [np.interp(COMMON_FPR, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)]
    )
    return np.mean(interp, axis=0), np.std(interp, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────────────────────────────────────


def _draw_panel(ax, panel_key, roc_curves, map_names, colors):
    for j, (method, method_data) in enumerate(roc_curves.items()):
        data = method_data.get(panel_key)
        if data is None or not data["fpr"]:
            continue

        n_runs = len(data["fpr"])
        mean_tpr, std_tpr = _mean_std_tpr(data["fpr"], data["tpr"])
        mean_auc = np.mean(data["auc"])
        std_auc = np.std(data["auc"])

        auc_str = f"{mean_auc:.2f} ± {std_auc:.2f}" if n_runs > 1 else f"{mean_auc:.2f}"
        label = f"{map_names.get(method, method)} (AUC {auc_str})"

        ax.plot(COMMON_FPR, mean_tpr, lw=2, color=colors[j], label=label)
        if n_runs > 1:
            ax.fill_between(
                COMMON_FPR,
                mean_tpr - std_tpr,
                mean_tpr + std_tpr,
                alpha=0.18,
                color=colors[j],
            )

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=21)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=21)
    ax.set_title(PANEL_TITLES[panel_key], fontsize=23)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(alpha=0.3)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        fancybox=True,
        handletextpad=0.5,
        labelspacing=0.7,
        fontsize=19,
    )


def _draw_zoom(ax, panel_key, zoom_type, roc_curves, colors):
    for j, (method, method_data) in enumerate(roc_curves.items()):
        data = method_data.get(panel_key)
        if data is None or not data["fpr"]:
            continue

        n_runs = len(data["fpr"])
        mean_tpr, std_tpr = _mean_std_tpr(data["fpr"], data["tpr"])

        ax.plot(COMMON_FPR, mean_tpr, lw=2, color=colors[j])
        if n_runs > 1:
            ax.fill_between(
                COMMON_FPR,
                mean_tpr - std_tpr,
                mean_tpr + std_tpr,
                alpha=0.18,
                color=colors[j],
            )

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax.grid(alpha=0.3)
    ax.set_title(ZOOM_TITLES[(panel_key, zoom_type)], fontsize=19)
    ax.set_xlabel("False Positive Rate", fontsize=17)
    ax.set_ylabel("True Positive Rate", fontsize=17)

    if zoom_type == "low_fpr":
        ax.set_xlim([0.0, 0.05])
        # ax.set_ylim([0.0, 1.00])
    else:  # high_tpr
        # ax.set_xlim([0.0, 0.20])
        ax.set_ylim([0.95, 1.002])

def plot_roc_curves(roc_curves, map_names):

    colors = sns.color_palette("Set1", n_colors=max(len(roc_curves), 3))

    fig = plt.figure(figsize=(22, 13))
    gs = gridspec.GridSpec(
        2,
        4,
        height_ratios=[3, 1],
        width_ratios=[1, 1, 1, 1],
        hspace=0.3,
    )
    gs = gridspec.GridSpec(
        2, 4, height_ratios=[3, 1], width_ratios=[1, 1, 1, 1], hspace=0.3
    )

    ax_holo = fig.add_subplot(gs[0, 0:2])
    ax_dyn = fig.add_subplot(gs[0, 2:4])
    ax_holo_low = fig.add_subplot(gs[1, 0])
    ax_holo_hi = fig.add_subplot(gs[1, 1])
    ax_dyn_low = fig.add_subplot(gs[1, 2])
    ax_dyn_hi = fig.add_subplot(gs[1, 3])

    _draw_panel(ax_holo, "midvholo", roc_curves, map_names, colors)
    _draw_panel(ax_dyn, "midvdyn", roc_curves, map_names, colors)
    _draw_zoom(ax_holo_low, "midvholo", "low_fpr", roc_curves, colors)
    _draw_zoom(ax_holo_hi, "midvholo", "high_tpr", roc_curves, colors)
    _draw_zoom(ax_dyn_low, "midvdyn", "low_fpr", roc_curves, colors)
    _draw_zoom(ax_dyn_hi, "midvdyn", "high_tpr", roc_curves, colors)

    # plt.suptitle("ROC Curve Comparison", fontsize=18, y=1.01)
    plt.tight_layout()
    return fig


def save_roc_plots(roc_curves, map_names, save_path="roc_curves_comparison.pdf"):
    """Render and save the 2×4 figure to disk."""
    fig = plot_roc_curves(roc_curves, map_names)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[ROC] saved -> {save_path}")


if __name__ == "__main__":
    # SET THIS: path to pouliquen.25.icdar project (contains notebooks/roc_folds/ from pouliquen.25.icdar)
    BASE = "midv-benchmark"
    # SET THIS: machine-specific DISTANTSERVER mount with baseline experiment artifacts
    DISTANTSERVER = "distantserverpath"

    # ── 1. Load WSL + MSM  (5 runs, one JSON per run) ─────────────────────────
    print("Loading WSL + MSM ...")
    wsl_msm_data = load_wsl_msm(
        {
            "fold_0": f"logs/onlyoriginsk0_full/plots/per_video/gmm/grouped_stats_video.json",
            "fold_1": f"logs/onlyoriginsk1_full/plots/per_video/gmm/grouped_stats_video.json",
            "fold_2": f"logs/onlyoriginsk2_full/plots/per_video/gmm/grouped_stats_video.json",
            "fold_3": f"logs/onlyoriginsk3_full/plots/per_video/gmm/grouped_stats_video.json",
            "fold_4": f"logs/onlyoriginsk4_full/plots/per_video/gmm/grouped_stats_video.json",
        }
    )

    # ── 2. Load HoloVerif  (5 runs, one JSON per run) ─────────────────────────
    print("Loading HoloVerif ...")
    prev_sota = {
        "allvideo_phd_holodetectorfinal_onlyorigins_5fps_s0_mobilevit_xxs": {
            "allvideo_phd_holodetectorfinal_onlyorigins_5fps_s0_mobilevit_xxs_k3": "mlflow_mlruns_path/425469025074057361/efba287561df497ba2b780086aea6a04/artifacts/roc_data/roc_curves_grouped.json",
            "allvideo_phd_holodetectorfinal_onlyorigins_5fps_s0_mobilevit_xxs_k1": "mlflow_mlruns_path/824739502828589041/e41721639f3844c388b15b66cd41b3fa/artifacts/roc_data/roc_curves_grouped.json",
            "allvideo_phd_holodetectorfinal_onlyorigins_5fps_s0_mobilevit_xxs_k2": "mlflow_mlruns_path/221226010727452916/fd42f1ce12bc40e29249c7e84d2fb214/artifacts/roc_data/roc_curves_grouped.json",
            "allvideo_phd_holodetectorfinal_onlyorigins_5fps_s0_mobilevit_xxs_k4": "mlflow_mlruns_path/162552195176991211/9c88562da8c04a32a1579768d6774c46/artifacts/roc_data/roc_curves_grouped.json",
            "allvideo_phd_holodetectorfinal_onlyorigins_5fps_s0_mobilevit_xxs_k0": "mlflow_mlruns_path/316820428512527962/21a416010b6c417d830400fa796cd096/artifacts/roc_data/roc_curves_grouped.json",
        },
        "allvideo_phd_wslf_old_15epochs_5fps_onlyorigins_norota_s0_mobilevit_xxs": {
            "allvideo_phd_wslf_old_15epochs_5fps_onlyorigins_norota_s0_mobilevit_xxs_k3": "mlflow_mlruns_path/425469025074057361/993d3052c8fd4eaebb16ac2284315ac7/artifacts/roc_data/roc_curves_grouped.json",
            "allvideo_phd_wslf_old_15epochs_5fps_onlyorigins_norota_s0_mobilevit_xxs_k1": "mlflow_mlruns_path/824739502828589041/b9eb98eecd2e4bf1a8eb8aea73d915b2/artifacts/roc_data/roc_curves_grouped.json",
            "allvideo_phd_wslf_old_15epochs_5fps_onlyorigins_norota_s0_mobilevit_xxs_k2": "mlflow_mlruns_path/221226010727452916/9a3a8b599269480a8656beb77ceb0e43/artifacts/roc_data/roc_curves_grouped.json",
            "allvideo_phd_wslf_old_15epochs_5fps_onlyorigins_norota_s0_mobilevit_xxs_k4": "mlflow_mlruns_path/162552195176991211/b7b6edfa686349dab029a2758a9c47a6/artifacts/roc_data/roc_curves_grouped.json",
            "allvideo_phd_wslf_old_15epochs_5fps_onlyorigins_norota_s0_mobilevit_xxs_k0": "mlflow_mlruns_path/316820428512527962/d47f82f11c354b54b68a471057c6e9d5/artifacts/roc_data/roc_curves_grouped.json",
        },
    }

    holoverif_data = load_holoverif(
        {
            "k0": "mlflow_mlruns_path/893369059486645024/c7c10b6a603c46c084032b579c23b8c2/artifacts/roc_data/roc_curves_grouped.json",
            "k1": "mlflow_mlruns_path/460682322732989539/787909bd4668469381f8aa3e0993e12f/artifacts/roc_data/roc_curves_grouped.json",
            "k2": "mlflow_mlruns_path/924518783309753644/e61bd025b75f435abcf4f54f15643add/artifacts/roc_data/roc_curves_grouped.json",
            "k3": "mlflow_mlruns_path/479828153994911896/0a9e3848c7d6401ca3d0fc06baa0c5ed/artifacts/roc_data/roc_curves_grouped.json",
            "k4": "mlflow_mlruns_path/703468346969507952/d04a3307f07b4f24be43a8ff847806a6/artifacts/roc_data/roc_curves_grouped.json",
        }
    )

    # ── 3. HoloVerif-SPAN  (5 folds, one JSON per fold) ───────────
    print("Loading HoloVerif-SPAN (GMM folds) ...")
    gmm_data = load_gmm_folds(
        {
            "fold_0": f"{BASE}/notebooks/roc_folds/fold_0_roc_curve.json",
            "fold_1": f"{BASE}/notebooks/roc_folds/fold_1_roc_curve.json",
            "fold_2": f"{BASE}/notebooks/roc_folds/fold_2_roc_curve.json",
            "fold_3": f"{BASE}/notebooks/roc_folds/fold_3_roc_curve.json",
            "fold_4": f"{BASE}/notebooks/roc_folds/fold_4_roc_curve.json",
        },
        # average_within_fold=True,
    )
    roc_curves = {name: load_holoverif(ps) for name, ps in prev_sota.items()}

    # ── 4. Assemble unified mapping ───────────────────────────────────────────
    roc_curves.update(
        {
            # "holoverif": holoverif_data,
            "gmm": gmm_data,
            "wsl_msm": wsl_msm_data,
        }
    )

    map_names = {
        "allvideo_phd_holodetectorfinal_onlyorigins_5fps_s0_mobilevit_xxs": "HoloVerif",
        "allvideo_phd_wslf_old_15epochs_5fps_onlyorigins_norota_s0_mobilevit_xxs": "WSL",
        # "holoverif": "HoloVerif (Trained with NON-LEGIT)",
        "gmm": "HoloVerif-Span",
        "wsl_msm": "MSM",
    }

    # ── 5. Render and save ────────────────────────────────────────────────────
    save_roc_plots(roc_curves, map_names, "roc_curves_comparison.pdf")
