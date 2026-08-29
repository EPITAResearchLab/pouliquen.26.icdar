"""export_latex.py — simplified
Reads grouped_stats_video_augmented.json across N folds and produces
LaTeX tables at every aggregation level (individual / merged / grouped).

Assumptions (post grouping.py fix):
  - 'grouped' section contains ALL groups including overlapping ones.
  - Each entry has: n_samples, auroc, ood_p95, ood_p99,
    anom_mean, anom_std, anom_median.
  - 'composition' maps group_name -> {fraud_name: n_samples}.
  - 'thresholds' holds {'p95': float, 'p99': float}.
"""
from __future__ import annotations

import json
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Configuration
# =============================================================================


FRAUD_GROUPS = {
    'static_midvholo': [
        'copy_without_holo', 'pseudo_holo_copy', 'photo_holo_copy'
    ],
    'photo_replacement': [
        'photo_replacement'
    ],
    'static_midvdynattack': [
        'no_holo', 'laser', 'plastified_led', 'plastified_lowreflect', 'plastified_noholo'
    ],
    'swap_midvdynattack': [
        'swap', 'swap_three'
    ],
    'dynamic_midvdynattack': [
        'holo_star_world', 'holo_completemask', 'leaf_holo', 'plain_holo', 'double_sticker'
    ],
    'midvdynattack': [
        'no_holo', 'laser', 'plastified_led', 'plastified_lowreflect', 'plastified_noholo', 'holo_star_world', 'holo_completemask', 'leaf_holo', 'plain_holo', 'double_sticker', 'swap', 'swap_three'
    ],
}

# Scalar metrics stored flat in each entry of the new JSON.
METRIC_KEYS = ["auroc", "ood_p95", "ood_p99"]

DISPLAY_NAMES: Dict[str, str] = {
    "static_midvholo":       r"\textsc{Static-MidvHolo}",
    "static_midvdynattack":  r"\textsc{Static-MidvDyn}",
    "dynamic_midvdynattack": r"\textsc{Dynamic-MidvDyn}",
    "swap_midvdynattack":    r"\textsc{Swap-MidvDyn}",
    "midvdynattack":         r"\textsc{MidvDyn (all)}",
    "photo_replacement":     r"\textsc{Photo-Replacement}",
    "copy_without_holo":     r"Copy w/o holo",
    "copy_with_holo":        r"Copy w/ holo",
    "no_holo":               r"No holo",
    "laser":                 r"Laser",
    "swap":                  r"Swap",
    "swap_three":            r"Swap (3-page)",
    "origins":               r"\textit{Origins (genuine)}",
}

GROUP_ORDER = [
    "static_midvholo",
    "photo_replacement",
    "static_midvdynattack",
    "swap_midvdynattack",
    "dynamic_midvdynattack",
    "midvdynattack",
]

OPERATING_POINT_METRICS = {
    "@99": "ood_p99",
    "@95": "ood_p95",
}


def display(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " ").title())


# =============================================================================
# Formatting helpers
# =============================================================================

def _fmt(value: Optional[float], decimals: int = 1, percent: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if percent:
        return f"{value * 100:.{decimals}f}"
    return f"{value:.{decimals}f}"


def _fmt_mean_std(mean: Optional[float], std: Optional[float] = None,
                  decimals: int = 1, percent: bool = False) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    scale = 100.0 if percent else 1.0
    m_str = f"{mean * scale:.{decimals}f}"
    if std is None or math.isnan(std):
        return m_str
    return rf"{m_str} $\pm$ {std * scale:.{decimals}f}"


def _bold(s: str) -> str:
    return r"\textbf{" + s + "}"


# =============================================================================
# Fold loading  —  mirrors the original process_all_folds_grouped
# =============================================================================

def _load_json(filepath: Path) -> Dict:
    """Load JSON, tolerating bare Infinity literals written by older code."""
    raw = filepath.read_text()
    raw = raw.replace(": Infinity", ": null").replace(":Infinity", ":null")
    return json.loads(raw)


def _extract_fold(data: Dict, section: str) -> Dict[str, Dict]:
    """
    Extract one fold's worth of per-name scalar metrics from a section
    ('individual', 'merged', or 'grouped').
    Returns  {name: {'n_samples': int, 'metrics': {key: float}}}.
    """
    fold: Dict[str, Dict] = {}
    for name, entry in data.get(section, {}).items():
        fold[name] = {
            "n_samples": entry.get("n_samples", 0),
            "metrics":   {k: entry.get(k, float("nan")) for k in METRIC_KEYS},
        }
    return fold


def _compute_statistics_across_folds(all_folds: List[Dict]) -> Dict:
    """
    Given a list of per-fold dicts  {name: {'n_samples', 'metrics'}},
    return  {name: {'n_samples', 'metrics': {key: {'mean', 'std', 'values'}}}}.
    Identical logic to the original compute_statistics_across_folds.
    """
    all_keys: set = set()
    for fd in all_folds:
        all_keys.update(fd.keys())

    stats: Dict = {}
    for key in all_keys:
        stats[key] = {"n_samples": None, "metrics": {}}
        for mk in METRIC_KEYS:
            values = []
            for fd in all_folds:
                if key in fd:
                    v = fd[key]["metrics"].get(mk, float("nan"))
                    if not math.isnan(v):
                        values.append(v)
                    if stats[key]["n_samples"] is None:
                        stats[key]["n_samples"] = fd[key]["n_samples"]
            stats[key]["metrics"][mk] = (
                {"mean": float(np.mean(values)),
                 "std":  float(np.std(values)),
                 "values": values}
                if values else None
            )
    return stats


def _extract_genuine_from_gmm(gmm_data: Dict) -> Dict:
    """
    Pull the 'origins' genuine row from gmm_results_video.json.
    Maps  ood_rates.origins.{p95,p99}  ->  {ood_p95, ood_p99}.
    AUROC for genuine -> NaN (rendered as '--').
    """
    ood = gmm_data.get("ood_rates", {}).get("origins", {})
    return {
        "origins": {
            "n_samples": ood.get("n", 0),
            "metrics": {
                "auroc":   float("nan"),
                "ood_p95": ood.get("p95", float("nan")),
                "ood_p99": ood.get("p99", float("nan")),
            },
        }
    }


def process_all_folds(
        base_path: str,
        n_folds: int,
        section: str = "merged",
        genuine_base_path: Optional[str] = None,
) -> Dict:
    """
    Load grouped_stats_video_augmented.json for each fold, extract `section`,
    optionally inject genuine (origins) row from gmm_results_video.json,
    then compute mean / std across folds.

    Drop-in replacement for the original process_all_folds_grouped with
    post_aggregate='none'  (grouping is done upstream in grouping.py now).
    """
    all_folds: List[Dict] = []

    for i in range(n_folds):
        data      = _load_json(Path(base_path.format(fold=i)))
        fold_data = _extract_fold(data, section)

        if genuine_base_path is not None:
            gmm_data = _load_json(Path(genuine_base_path.format(fold=i)))
            fold_data.update(_extract_genuine_from_gmm(gmm_data))

        all_folds.append(fold_data)

    return _compute_statistics_across_folds(all_folds)


# =============================================================================
# Shared table helpers
# =============================================================================

def _metric_cell(stats: Dict, name: str, mk: str, *,
                 percent: bool = False, bold: bool = False) -> str:
    entry = stats.get(name, {}).get("metrics", {}).get(mk)
    if entry is None:
        return "--"
    s = _fmt_mean_std(entry["mean"], entry["std"], percent=percent)
    return _bold(s) if bold else s


def _save_tables(tables: Dict[str, str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerated tables:")
    for name, content in tables.items():
        filepath = output_dir / f"{name}.tex"
        filepath.write_text(content)
        print(f"  - {filepath}")
        print(content)
        print("\n" + "=" * 80 + "\n")

    combined = output_dir / "all_tables.tex"
    combined.write_text(
        "% Auto-generated LaTeX tables\n% Requires: booktabs\n\n"
        + "\n\n".join(f"% === {n} ===\n{c}" for n, c in tables.items())
    )
    print(f"\nCombined file: {combined}")


# =============================================================================
# Table: detailed  (one row per fraud, group-avg summary rows)
# =============================================================================

def generate_latex_table_detailed(
        stats_fraud: Dict,
        caption: str,
        label: str,
        include_genuine: bool = True,
) -> str:
    """
    Detailed table: one row per fraud type within each group,
    followed by a bold n_samples-weighted summary row per group.
    Reads from 'merged'-level stats.
    """
    # Discover groups from DISPLAY_NAMES order, members from stats keys

    col_spec = r"l r c c c"
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\vspace{-2mm}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"Attack Type & $N$ & AUROC & OOD$_{95}$ & OOD$_{99}$ \\\\",
        r"\midrule",
    ]

    if include_genuine and "origins" in stats_fraud:
        n = stats_fraud["origins"]["n_samples"] or ""
        auroc = _metric_cell(stats_fraud, "origins", "auroc")
        p95   = _metric_cell(stats_fraud, "origins", "ood_p95", percent=True)
        p99   = _metric_cell(stats_fraud, "origins", "ood_p99", percent=True)
        label_cell = _bold(display("origins"))
        lines.append(rf"{label_cell} & {n} & {auroc} & {p95} & {p99} \\")
        lines.append(r"\midrule")

    for group_name in GROUP_ORDER:
        members = [f for f in FRAUD_GROUPS.get(group_name, []) if f in stats_fraud]
        if not members:
            continue

        lines.append(
            rf"\multicolumn{{5}}{{l}}{{\textit{{{display(group_name)}}}}} \\\\"
        )

        ns, p95s, p99s, aurocs = [], [], [], []
        for fraud in members:
            e  = stats_fraud[fraud]
            n  = e["n_samples"] or ""
            ns.append(e["n_samples"] or 0)
            auroc = _metric_cell(stats_fraud, fraud, "auroc")
            p95   = _metric_cell(stats_fraud, fraud, "ood_p95", percent=True)
            p99   = _metric_cell(stats_fraud, fraud, "ood_p99", percent=True)
            aurocs.append(e["metrics"]["auroc"]["mean"] if e["metrics"]["auroc"] else float("nan"))
            p95s.append(e["metrics"]["ood_p95"]["mean"] if e["metrics"]["ood_p95"] else float("nan"))
            p99s.append(e["metrics"]["ood_p99"]["mean"] if e["metrics"]["ood_p99"] else float("nan"))
            lines.append(rf"  {display(fraud)} & {n} & {auroc} & {p95} & {p99} \\")

        # weighted group-average summary row
        total_n = sum(ns)
        def wavg(vals):
            pairs = [(v, w) for v, w in zip(vals, ns) if not math.isnan(v)]
            return sum(v*w for v,w in pairs)/sum(w for _,w in pairs) if pairs else float("nan")

        lines.append(r"  \cmidrule(lr){1-5}")
        lines.append(
            rf"  {_bold(display(group_name))} & {_bold(str(total_n))} "
            rf"& {_bold(_fmt(wavg(aurocs), decimals=1))} "
            rf"& {_bold(_fmt(wavg(p95s), decimals=1, percent=True))} "
            rf"& {_bold(_fmt(wavg(p99s), decimals=1, percent=True))} \\\\"
        )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# =============================================================================
# Table: main results  (one row per group)
# =============================================================================

def generate_latex_table_by_group(
        stats_grouped: Dict,
        caption: str,
        label: str,
        stats_fraud: Optional[Dict] = None,
) -> str:
    """
    One row per group. Reads directly from 'grouped'-level stats
    (produced correctly by grouping.py after the FRAUD_TO_GROUPS fix).
    stats_fraud is accepted for the origins row only.
    """
    col_spec = r"l r c c c"
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\vspace{-2mm}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"Category & $N$ & AUROC & OOD$_{95}$ & OOD$_{99}$ \\\\",
        r"\midrule",
    ]

    src = stats_fraud or {}
    if "origins" in src:
        n     = src["origins"]["n_samples"] or ""
        auroc = _metric_cell(src, "origins", "auroc")
        p95   = _metric_cell(src, "origins", "ood_p95", percent=True)
        p99   = _metric_cell(src, "origins", "ood_p99", percent=True)
        lines.append(rf"{_bold(display('origins'))} & {n} & {auroc} & {p95} & {p99} \\")
        lines.append(r"\midrule")

    best = {mk: float("inf") for mk in ["auroc", "ood_p95", "ood_p99"]}
    for gname in GROUP_ORDER:
        if gname not in stats_grouped:
            continue
        for mk in best:
            e = stats_grouped[gname]["metrics"].get(mk)
            if e:
                best[mk] = min(best[mk], e["mean"])

    for gname in GROUP_ORDER:
        if gname not in stats_grouped:
            continue
        n     = stats_grouped[gname]["n_samples"] or ""
        auroc = _metric_cell(stats_grouped, gname, "auroc",   bold=(stats_grouped[gname]["metrics"].get("auroc")  or {}).get("mean") == best["auroc"])
        p95   = _metric_cell(stats_grouped, gname, "ood_p95", percent=True, bold=(stats_grouped[gname]["metrics"].get("ood_p95") or {}).get("mean") == best["ood_p95"])
        p99   = _metric_cell(stats_grouped, gname, "ood_p99", percent=True, bold=(stats_grouped[gname]["metrics"].get("ood_p99") or {}).get("mean") == best["ood_p99"])
        lines.append(rf"{display(gname)} & {n} & {auroc} & {p95} & {p99} \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# =============================================================================
# Table: operating points  (@99 / @95 recall per group)
# =============================================================================

def generate_latex_table_operating_points(
        stats_fraud: Dict,
        caption: str,
        label: str,
        model_name: str = "HoloVerif-Span",
) -> str:
    """
    One row per operating point (@99 / @95), one column per group.
    N and weighted-avg OOD computed from 'merged'-level stats_fraud.
    origins column = genuine FPR from gmm_results_video.json.
    """

    present_groups = [g for g in GROUP_ORDER
                      if any(f in stats_fraud for f in FRAUD_GROUPS.get(g, []))]

    col_spec = "ll" + "r" * (1 + len(present_groups))
    col_headers = (
        [r"\textbf{origins}"]
        + [rf"\textbf{{{display(g)}}}" for g in present_groups]
    )

    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\renewcommand{\arraystretch}{1.2}", r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Metric} & " + " & ".join(col_headers) + r" \\\\",
        r"\midrule",
    ]

    for op_tag, mk in OPERATING_POINT_METRICS.items():
        origins_cell = _metric_cell(stats_fraud, "origins", mk, percent=True)

        group_cells = []
        for g in present_groups:
            members = [f for f in FRAUD_GROUPS.get(g, []) if f in stats_fraud]
            ns      = [stats_fraud[f]["n_samples"] or 0 for f in members]
            # per-fold scalar lists, one entry per fold
            fold_vals = [
                (stats_fraud[f]["metrics"][mk] or {}).get("values", [])
                for f in members
            ]
            # for each fold, compute a n_samples-weighted mean across members
            n_folds = max((len(fv) for fv in fold_vals), default=0)
            fold_means = []
            for i in range(n_folds):
                pairs = [
                    (fold_vals[j][i], ns[j])
                    for j in range(len(members))
                    if i < len(fold_vals[j]) and not math.isnan(fold_vals[j][i])
                ]
                if pairs:
                    fold_means.append(sum(v*w for v,w in pairs) / sum(w for _,w in pairs))

            if fold_means:
                group_cells.append(_fmt_mean_std(
                    float(np.mean(fold_means)),
                    float(np.std(fold_means)),
                    percent=True,
                ))
            else:
                group_cells.append("--")

        lines.append(
            f"{model_name}{op_tag} & recall & {origins_cell} & "
            + " & ".join(group_cells) + r" \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# =============================================================================
# Entry point  —  identical call signature to original main()
# =============================================================================

def generate_detailed_video_from_grouped_v2(
        base_path: str,
        n_folds: int,
        output_dir,
        genuine_base_path: Optional[str] = None,
        model_name: str = "HoloVerif-Span",
) -> None:
    """
    Main entry point — identical to the original.
    Generates three tables:
      - detailed_video.tex   : one row per fraud type + group-avg rows
      - main_results.tex     : one row per category
      - operating_points.tex : @99 / @95 recall per category
    """
    output_dir = Path(output_dir)
    print("Processing grouped_stats_video_augmented.json data...")

    # 'merged' section: per-fraud stats averaged across folds
    stats_fraud = process_all_folds(
        base_path, n_folds,
        section="merged",
        genuine_base_path=genuine_base_path,
    )

    # 'grouped' section: pre-computed group aggregates (all groups present now)
    stats_grouped = process_all_folds(
        base_path, n_folds,
        section="grouped",
        genuine_base_path=None,   # origins lives in merged
    )

    has_genuine = genuine_base_path is not None and "origins" in stats_fraud

    tables = {
        "detailed_video": generate_latex_table_detailed(
            stats_fraud,
            caption=(
                "Detailed video-level fraud detection results for each attack type. "
                "$N$ denotes the number of videos. "
                + (
                    "The \\textit{Genuine} row reports the false-positive rate (FPR) "
                    "at the p95/p99 thresholds calibrated on the validation set. "
                    if has_genuine else ""
                )
                + "Best results per category in \\textbf{bold}."
            ),
            label="tab:detailed_video",
            include_genuine=has_genuine,
        ),
        "main_results": generate_latex_table_by_group(
            stats_grouped,
            caption=(
                "Video-level fraud detection results across attack categories. "
                "OOD rates at 95\\% and 99\\% thresholds, and AUROC. "
                f"Results averaged over {n_folds} folds (mean $\\pm$ std). "
                "Best per metric in \\textbf{bold}."
            ),
            label="tab:main_results",
            stats_fraud=stats_fraud,
        ),
        "operating_points": generate_latex_table_operating_points(
            stats_fraud=stats_fraud,
            caption=(
                f"Recall at operating points @99 and @95 "
                f"(mean $\\pm$ std over {n_folds} folds). "
                "Thresholds are calibrated on the validation set. "
                "The \\textit{origins} column is the false-positive rate on the test set."
            ),
            label="tab:operating_points",
            model_name=model_name,
        ),
    }

    _save_tables(tables, output_dir)


# kept for backward compatibility
def generate_detailed_video_from_grouped(
        base_path: str,
        n_folds: int,
        output_dir: Path,
        genuine_base_path: Optional[str] = None,
) -> None:
    generate_detailed_video_from_grouped_v2(
        base_path, n_folds, output_dir,
        genuine_base_path=genuine_base_path,
    )


def main() -> None:
    name       = "onlyorigins"
    MODEL_NAME = "HoloVerif-Span"

    BASE_PATH_NEW = f"logs/{name}" + "k{fold}_full/plots/per_video/gmm/grouped_stats_video.json"
    GENUINE_PATH  = f"logs/{name}" + "k{fold}_full/plots/per_video/gmm/gmm_results_video.json"

    generate_detailed_video_from_grouped_v2(
        BASE_PATH_NEW,
        n_folds=5,
        output_dir=f"latex_tables/{name}",
        genuine_base_path=GENUINE_PATH,
        model_name=MODEL_NAME,
    )

    print("\n" + "=" * 80)
    print("LaTeX preamble requirements:")
    print(r"""
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
""")


if __name__ == "__main__":
    main()
