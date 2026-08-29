"""compare_methods.py

Load per-fold ROC JSON files for N methods (all in the canonical holoverif
schema), compute per-group AUC mean ± std across folds, and emit the
paper LaTeX table.

Canonical JSON schema  (one file = one fold / one run)
------------------------------------------------------
{
    "groups": {
        "<group_key>": {
            "fpr": [float, ...],   # flat 1-D array  OR  list of 1-D arrays
            "tpr": [float, ...],
            "auc": float           # scalar  OR  list[float]
        },
        ...
    }
}

Usage
-----
Edit METHOD_CONFIGS at the bottom and run:

    python compare_methods.py [--out table.tex] [--csv results.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Column schema
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Maps raw group names in the JSON to the 6 canonical columns.
GROUP_ALIASES: Dict[str, str] = {
    # - Vanilla / MIDV-Holo -
    "vanilla": "vanilla",
    "midvholo": "vanilla",
    "holo": "vanilla",
    "MIDV_Holo": "vanilla",
    # - Photo-replacement -
    "photorep": "photorep",
    "photo_rep": "photorep",
    "photoreplacement": "photorep",
    # - Static template -
    "static": "static",
    "midvdyn_static": "static",
    "static template": "static",
    # - Static-swap -
    "staticswap": "staticswap",
    "static_swap": "staticswap",
    "midvdyn_staticswap": "staticswap",
    "swap template": "staticswap",
    "static template swaping": "staticswap",
    # - Dynamic -
    "dynamic": "dynamic",
    "midvdyn_dynamic": "dynamic",
    "dynamic template": "dynamic",
    # - Combined / Mix -
    "combined": "combined",
    "mix": "combined",
    "midvdyn": "combined",  # full dynattack bucket -> mix
}

COLUMN_KEYS: List[str] = [
    "vanilla",
    "photorep",
    "static",
    "staticswap",
    "dynamic",
    "combined",
]
COLUMN_LABELS: Dict[str, str] = {
    "vanilla": r"\textbf{Vanilla}",
    "photorep": r"\textbf{Photo Rep.}",
    "static": r"\textbf{Static}",
    "staticswap": r"\textbf{Static-swap}",
    "dynamic": r"\textbf{Dynamic}",
    "combined": r"\textbf{Mix}",
}
COLUMN_VIDS: Dict[str, str] = {
    "vanilla": "60 vids*",
    "photorep": "20 vids",
    "static": "110 vids",
    "staticswap": "40 vids",
    "dynamic": "90 vids",
    "combined": "380 vids",
}


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Loader  - holoverif JSON schema
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def _extract_auc(g_data: dict) -> List[float]:
    """
    Extract AUC value(s) from a group entry.

    The schema allows three forms:
      - auc: float                 (single fold file, single run)
      - auc: [float, ...]          (single fold file, multiple dataset-runs averaged)
      - fpr/tpr present but no auc (compute from trapezoidal rule on first pair)

    Returns a flat list of AUC floats (one per dataset-run in this fold).
    """
    auc_raw = g_data.get("auc")
    if auc_raw is None:
        # Fallback: compute from the stored fpr/tpr
        fpr = g_data.get("fpr", [])
        tpr = g_data.get("tpr", [])
        if not fpr or not tpr:
            return []
        # Handle list-of-arrays vs flat array
        if isinstance(fpr[0], (list, np.ndarray)):
            return [float(np.trapz(np.array(t), np.array(f))) for f, t in zip(fpr, tpr)]
        return [float(np.trapz(np.array(tpr), np.array(fpr)))]

    if isinstance(auc_raw, (int, float)):
        return [float(auc_raw)]
    # list form
    return [float(v) for v in auc_raw]


def load_method_folds(
    fold_paths,  # str | Path | list | {label: path}
    group_aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, List[float]]:
    """
    Load one method’s fold files and return:
        group_key -> [auc_run0, auc_run1, ...]  (one per fold file / dataset-run)

    Each fold file is treated as one independent measurement.
    Within a single fold file, multiple dataset-runs belonging to
    the same canonical column are AVERAGED before being appended,
    so 5 fold files always yield exactly 5 values per column.

    Parameters
    ----------
    fold_paths : str | Path | list | {label: path}
        One or more JSON paths (one per fold).
    group_aliases : dict, optional
        Override for GROUP_ALIASES.
    """
    if group_aliases is None:
        group_aliases = GROUP_ALIASES

    if isinstance(fold_paths, dict):
        paths = list(fold_paths.values())
    elif isinstance(fold_paths, (str, Path)):
        paths = [fold_paths]
    else:
        paths = list(fold_paths)

    result: Dict[str, List[float]] = defaultdict(list)

    for path in paths:
        with open(path) as f:
            manifest = json.load(f)

        # Accumulate all dataset-runs per canonical column within this fold.
        fold_bucket: Dict[str, List[float]] = defaultdict(list)
        for raw_name, g_data in manifest.get("groups", {}).items():
            col = group_aliases.get(raw_name)
            if col is None:
                # Try lowercase / stripped
                col = group_aliases.get(
                    raw_name.lower().replace("-", "").replace("_", "")
                )
            if col is None:
                continue  # unknown group - skip silently
            for auc in _extract_auc(g_data):
                fold_bucket[col].append(auc)

        # One value per column per fold = mean of dataset-runs within the fold.
        for col, aucs in fold_bucket.items():
            result[col].append(float(np.mean(aucs)))

    return dict(result)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Statistics
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def _fmt(mean: float, std: float, n: int) -> str:
    """Return '93.4 ± 1.9' (×100, 1 dp).  '--' if no data."""
    if n == 0:
        return "--"
    if n == 1:
        return f"{mean * 100:.1f}"
    return f"{mean * 100:.1f} $\\pm$ {std * 100:.1f}"


def summarise(
    group_aucs: Dict[str, List[float]],
) -> Dict[str, Tuple[float, float, int]]:
    """Return {col_key: (mean, std, n)} for each column."""
    out = {}
    for col in COLUMN_KEYS:
        vals = group_aucs.get(col, [])
        n = len(vals)
        mean = float(np.mean(vals)) if n else 0.0
        std = float(np.std(vals)) if n > 1 else 0.0
        out[col] = (mean, std, n)
    return out


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LaTeX emitter
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def latex_table(
    methods: Dict[str, Dict[str, List[float]]],  # display_name -> {col: [auc, ...]}
    sections: Optional[List[Tuple[str, List[str]]]] = None,
    caption: str = (
        r"Experimental results on \MIDVHOLO and \MIDVDYNATTACK datasets. "
        r"* AUC was computed over 5 folds."
    ),
    label: str = "tab:resultsfull",
) -> str:
    """
    Build the full LaTeX table string.

    Parameters
    ----------
    methods : OrderedDict  display_name -> {col_key: [auc_values]}
    sections : list of (section_title, [display_name, ...]) groupings.
               If None, all methods are placed in a single unnamed section.
    """
    ncols = len(COLUMN_KEYS)

    lines: List[str] = []
    A = lines.append  # shorthand

    A(r"\begin{table}[tbp]")
    A(r"    \centering")
    A(rf"    \caption{{{caption}}}")
    A(rf"    \label{{{label}}}")
    A(r"    \resizebox{\textwidth}{!}{%")
    A(r"    \begin{tabular}{l" + "c" * ncols + "}")
    A(r"        \toprule")

    # ─ Header rows ---------------------------------------------------------------------------------------------------------------------------------------------------
    A(
        r"        & \multicolumn{2}{c}{\textbf{\MIDVHOLO} (test split)}"
        r" & \multicolumn{3}{c}{\textbf{MIDV-DynAttack} (full)}"
        r" & \textbf{Both} \\"
    )
    col_label_str = " & ".join(COLUMN_LABELS[c] for c in COLUMN_KEYS)
    A(f"        & {col_label_str} \\\\")
    vid_str = " & ".join(COLUMN_VIDS[c] for c in COLUMN_KEYS)
    A(
        r"        \multirow{3}{*}{\begin{tabular}[c]{@{}l@{}}"
        r"\textbf{Verification}\\\\\textbf{methods $\downarrow$}\end{tabular}}"
    )
    A(f"        & {vid_str} \\\\")
    cmidrules = " ".join(rf"\cmidrule(lr){{{i + 2}-{i + 2}}}" for i in range(ncols))
    A(f"        {cmidrules}")
    auc_header = " & ".join([r"\textbf{AUC}"] * ncols)
    A(f"        & {auc_header} \\\\")
    A(r"        \midrule")

    # ─ Data rows ----------------------------------------------------------------------------------------------------------------------------------------------------------─
    def _row(display_name: str, group_aucs: Dict[str, List[float]]) -> str:
        stats = summarise(group_aucs)
        cells = " & ".join(_fmt(*stats[c]) for c in COLUMN_KEYS)
        return f"        {display_name} & {cells} \\\\"

    if sections is None:
        # No sections - just dump all methods
        for name, group_aucs in methods.items():
            A(_row(name, group_aucs))
    else:
        for sec_title, names in sections:
            A(rf"        \midrule")
            A(rf"        \multicolumn{{{ncols + 1}}}{{c}}{{\textit{{{sec_title}}}}} \\")
            A(r"        \midrule")
            for name in names:
                if name in methods:
                    A(_row(name, methods[name]))
                else:
                    print(f"[WARN] method '{name}' not found in data", flush=True)

    A(r"        \bottomrule")
    A(r"    \end{tabular}")
    A(r"    } %")
    A(r"\end{table}")

    return "\n".join(lines)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CSV helper
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def csv_table(methods: Dict[str, Dict[str, List[float]]]) -> str:
    import io

    buf = io.StringIO()
    header = (
        ["method"]
        + [f"{c}_mean" for c in COLUMN_KEYS]
        + [f"{c}_std" for c in COLUMN_KEYS]
    )
    buf.write(",".join(header) + "\n")
    for name, group_aucs in methods.items():
        stats = summarise(group_aucs)
        row = [name]
        for c in COLUMN_KEYS:
            m, s, n = stats[c]
            row.append(f"{m * 100:.2f}" if n else "")
        for c in COLUMN_KEYS:
            m, s, n = stats[c]
            row.append(f"{s * 100:.2f}" if n > 1 else "")
        buf.write(",".join(row) + "\n")
    return buf.getvalue()


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TODO Config - edit here TODO
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# SET THIS: absolute path to this repository root (used for sequenceclassif/jsonres/ paths below)
BASE = "base/repo/path"  # unused placeholder
SECONDREPO = "second/repo/path"
# SET THIS: absolute path to the mlruns directory from pouliquen.25.icdar experiments
MLRUNS = f"{SECONDREPO}/path/to/mlruns"


# Each entry:  display_name  ->  {fold_label: path, ...}
# Every path must be a holoverif-schema JSON.
METHOD_CONFIGS: Dict[str, Dict[str, str]] = {
    #  SOTA: Trained on Legits + Non-Legits -----------------------------------------------------------------------------
    r"\MIDVHOLOMETHOD": {
        "k0": f"{MLRUNS}/316820428512527962/21a416010b6c417d830400fa796cd096/artifacts/roc_data/roc_curves_grouped.json",
        "k1": f"{MLRUNS}/824739502828589041/787909bd4668469381f8aa3e0993e12f/artifacts/roc_data/roc_curves_grouped.json",
        "k2": f"{MLRUNS}/221226010727452916/fd42f1ce12bc40e29249c7e84d2fb214/artifacts/roc_data/roc_curves_grouped.json",
        "k3": f"{MLRUNS}/425469025074057361/efba287561df497ba2b780086aea6a04/artifacts/roc_data/roc_curves_grouped.json",
        "k4": f"{MLRUNS}/162552195176991211/9c88562da8c04a32a1579768d6774c46/artifacts/roc_data/roc_curves_grouped.json",
    },
    r"mvit-\WSLMETHOD (Legits+Non-Legits)": {
        "k0": f"{MLRUNS}/316820428512527962/07cecb4962b645b0a2a75441e01e7cf4/artifacts/roc_data/roc_curves_grouped.json",
        "k1": f"{MLRUNS}/824739502828589041/26f8342b7749497990c13261bd13ae0d/artifacts/roc_data/roc_curves_grouped.json",
        "k2": f"{MLRUNS}/221226010727452916/b5c6774d6ef648068b036857fbe9b32f/artifacts/roc_data/roc_curves_grouped.json",
        "k3": f"{MLRUNS}/425469025074057361/58b1bcf7721a422ea86f544695541050/artifacts/roc_data/roc_curves_grouped.json",
        "k4": f"{MLRUNS}/162552195176991211/596b26b677a8412f87ad1b81f7522efc/artifacts/roc_data/roc_curves_grouped.json",
    },
    r"mvit-\ADVANCEDCLASSIFERMETHOD (Legits+Non-Legits)": {
        "k0": f"{MLRUNS}/316820428512527962/a8b91857d90b4b989c596016ee47f18a/artifacts/roc_data/roc_curves_grouped.json",
        "k1": f"{MLRUNS}/824739502828589041/5b96a0666b0940d589df70b9c23b1a69/artifacts/roc_data/roc_curves_grouped.json",
        "k2": f"{MLRUNS}/221226010727452916/213e35e99cdd46c19a426b2c84702e1c/artifacts/roc_data/roc_curves_grouped.json",
        "k3": f"{MLRUNS}/425469025074057361/cb1b624272004092902bc0d4755a4396/artifacts/roc_data/roc_curves_grouped.json",
        "k4": f"{MLRUNS}/162552195176991211/41b60748d71b4de2a1821aa7c072ce93/artifacts/roc_data/roc_curves_grouped.json",
    },
    #  SOTA: Trained only on Legits -------------------------------------------------------------------------------------------─
    r"mvit-\WSLMETHOD (Legits only)": {
        "k0": f"{MLRUNS}/316820428512527962/d47f82f11c354b54b68a471057c6e9d5/artifacts/roc_data/roc_curves_grouped.json",
        "k1": f"{MLRUNS}/824739502828589041/b9eb98eecd2e4bf1a8eb8aea73d915b2/artifacts/roc_data/roc_curves_grouped.json",
        "k2": f"{MLRUNS}/221226010727452916/9a3a8b599269480a8656beb77ceb0e43/artifacts/roc_data/roc_curves_grouped.json",
        "k3": f"{MLRUNS}/425469025074057361/993d3052c8fd4eaebb16ac2284315ac7/artifacts/roc_data/roc_curves_grouped.json",
        "k4": f"{MLRUNS}/162552195176991211/b7b6edfa686349dab029a2758a9c47a6/artifacts/roc_data/roc_curves_grouped.json",

    },
    r"mvit-\ADVANCEDCLASSIFERMETHOD (Legits only)": {
        "k0": f"{MLRUNS}/316820428512527962/21a416010b6c417d830400fa796cd096/artifacts/roc_data/roc_curves_grouped.json",
        "k1": f"{MLRUNS}/824739502828589041/e41721639f3844c388b15b66cd41b3fa/artifacts/roc_data/roc_curves_grouped.json",
        "k2": f"{MLRUNS}/221226010727452916/fd42f1ce12bc40e29249c7e84d2fb214/artifacts/roc_data/roc_curves_grouped.json",
        "k3": f"{MLRUNS}/425469025074057361/efba287561df497ba2b780086aea6a04/artifacts/roc_data/roc_curves_grouped.json",
        "k4": f"{MLRUNS}/162552195176991211/9c88562da8c04a32a1579768d6774c46/artifacts/roc_data/roc_curves_grouped.json",
    },
    #  Proposed methods (Legits only) ------------------------------------------------------------------------------------
    r"mvit-\WSLMETHOD + MSM + GMM": {
        "k0": f"path/to/logs/onlyoriginsk0_full/plots/per_video/gmm/grouped_stats_video.json",
        "k1": f"path/to/logs/onlyoriginsk1_full/plots/per_video/gmm/grouped_stats_video.json",
        "k2": f"path/to/logs/onlyoriginsk2_full/plots/per_video/gmm/grouped_stats_video.json",
        "k3": f"path/to/logs/onlyoriginsk3_full/plots/per_video/gmm/grouped_stats_video.json",
        "k4": f"path/to/logs/onlyoriginsk4_full/plots/per_video/gmm/grouped_stats_video.json",
    },
    r"videomae-\ADVANCEDCLASSIFERMETHOD Span": {
        "k0": f"{SECONDREPO}/sequenceclassif/jsonres/k0-phd2_finalfinal_origins_full_test.json",
        "k1": f"{SECONDREPO}/sequenceclassif/jsonres/k1-phd2_finalfinal_origins_full_test.json",
        "k2": f"{SECONDREPO}/sequenceclassif/jsonres/k2-phd2_finalfinal_origins_full_test.json",
        "k3": f"{SECONDREPO}/sequenceclassif/jsonres/k3-phd2_finalfinal_origins_full_test.json",
        "k4": f"{SECONDREPO}/sequenceclassif/jsonres/k4-phd2_finalfinal_origins_full_test.json",
    },
}

# Table section groupings  (section title, [display_names in order])
SECTIONS = [
    (
        r"SOTA methods - Trained/Calibrated on \textbf{Legits + Non-Legits}",
        [
            r"\MIDVHOLOMETHOD",
            r"mvit-\WSLMETHOD (Legits+Non-Legits)",
            r"mvit-\ADVANCEDCLASSIFERMETHOD (Legits+Non-Legits)",
        ],
    ),
    (
        r"SOTA methods - Trained only on \textbf{Legits}",
        [
            r"mvit-\WSLMETHOD (Legits only)",
            r"mvit-\ADVANCEDCLASSIFERMETHOD (Legits only)",
        ],
    ),
    (
        r"Proposed methods - Trained only on \textbf{Legits}",
        [
            r"mvit-\WSLMETHOD + MSM + GMM",
            r"videomae-\ADVANCEDCLASSIFERMETHOD Span",
        ],
    ),
]


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Entry-point
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build LaTeX AUC comparison table.")
    parser.add_argument(
        "--out", default="table.tex", help="Output .tex file (default: table.tex)"
    )
    parser.add_argument("--csv", default=None, help="Also write a CSV summary")
    parser.add_argument(
        "--print", dest="print_", action="store_true", help="Print LaTeX to stdout"
    )
    args = parser.parse_args(argv)

    # Load all methods
    print("Loading methods ...")
    loaded: Dict[str, Dict[str, List[float]]] = {}
    for name, fold_paths in METHOD_CONFIGS.items():
        try:
            loaded[name] = load_method_folds(fold_paths)
            n_folds = max((len(v) for v in loaded[name].values()), default=0)
            print(f"  OK  '{name}'  ({n_folds} folds)")
        except FileNotFoundError as e:
            print(f"  SKIP '{name}'  - {e}")

    # Emit LaTeX
    tex = latex_table(loaded, sections=SECTIONS)
    with open(args.out, "w") as f:
        f.write(tex)
    print(f"\nLaTeX table written to: {args.out}")

    if args.print_:
        print("\n" + tex)

    if args.csv:
        csv = csv_table(loaded)
        with open(args.csv, "w") as f:
            f.write(csv)
        print(f"CSV written to: {args.csv}")


if __name__ == "__main__":
    main()
