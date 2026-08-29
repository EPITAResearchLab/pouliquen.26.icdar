"""
eval/threshold_detector.py

Interpretable two-threshold anomaly detector.

Logic
-----
Signal 1 – seq_cosine   : LOW  values → fake/OOD.   Deviation = max(0, thr_bot - x) / val_std
Signal 2 – recon_cosine : HIGH values → anomalous.   Deviation = max(0, x - thr_top) / val_std
Combined anomaly score  = elementwise max(dev_seq, dev_recon)   [OR-logic]

Thresholds fitted exclusively on the validation set.
All metrics computed on the test set (never on val).

Public API
----------
ThresholdDetector             – fit / score / predict
compute_threshold_auroc       – single-call wrapper (one level)
compute_stats_all_levels      – fit + evaluate at individual / merged / grouped levels
print_stats_summary           – pretty-print all three aggregation levels
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    warnings.warn("scikit-learn not found – AUROC/ROC metrics will be skipped.")

from .grouping import (
    merge_id_passport_variants,
    aggregate_to_groups,
    get_group_composition,
    FRAUD_GROUPS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Internals
# ══════════════════════════════════════════════════════════════════════════════

def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if not _SKLEARN_OK or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_roc_curve(y_true, y_score):
    if not _SKLEARN_OK or len(np.unique(y_true)) < 2:
        e = np.array([])
        return e, e, e
    return roc_curve(y_true, y_score)


def _to_list(arr) -> list:
    return arr.tolist() if isinstance(arr, np.ndarray) else list(arr)


def _extract(d: dict, key: str) -> np.ndarray:
    if key not in d:
        raise KeyError(f"Expected key '{key}' in scores dict, got: {list(d.keys())}")
    return np.asarray(d[key], dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# Core detector
# ══════════════════════════════════════════════════════════════════════════════

class ThresholdDetector:
    """
    Two-threshold OR-logic detector.

    Parameters
    ----------
    q_bottom : lower quantile on val seq_cosine   → bottom threshold
    q_high   : upper quantile on val recon_cosine → top threshold

    After fit(), the following attributes are set:
        thr_bot, thr_top          – raw threshold values
        std_seq, std_recon        – val std for normalisation
        p95_combined, p99_combined – combined-score percentiles on val
    """

    def __init__(self, q_bottom: float = 0.05, q_high: float = 0.95):
        self.q_bottom = q_bottom
        self.q_high   = q_high
        # set by fit()
        self.thr_bot: float = float("nan")
        self.thr_top: float = float("nan")
        self.std_seq: float = 1.0
        self.std_recon: float = 1.0
        self.p95_combined: float = float("nan")
        self.p99_combined: float = float("nan")

    # ──────────────────────────────────────────────────────────────
    # Fit (validation set only)
    # ──────────────────────────────────────────────────────────────

    def fit(self, val_scores: Dict[str, np.ndarray]) -> "ThresholdDetector":
        """
        Fit thresholds from validation set.

        Parameters
        ----------
        val_scores : dict with keys 'seq_cosine' and 'recon_cosine'
        """
        seq   = _extract(val_scores, "seq_cosine")
        recon = _extract(val_scores, "recon_cosine")

        self.thr_bot   = float(np.percentile(seq,   self.q_bottom * 100))
        self.thr_top   = float(np.percentile(recon, self.q_high   * 100))
        self.std_seq   = float(seq.std())   or 1.0
        self.std_recon = float(recon.std()) or 1.0

        # lock val combined scores for OOD-rate percentiles
        val_combined = self._combined(seq, recon)
        self.p95_combined = float(np.percentile(val_combined, 95))
        self.p99_combined = float(np.percentile(val_combined, 99))
        return self

    # ──────────────────────────────────────────────────────────────
    # Scoring helpers
    # ──────────────────────────────────────────────────────────────

    def _dev_seq(self, seq: np.ndarray) -> np.ndarray:
        """Normalised downward deviation of seq_cosine below thr_bot."""
        return np.maximum(0.0, self.thr_bot - seq) / self.std_seq

    def _dev_recon(self, recon: np.ndarray) -> np.ndarray:
        """Normalised upward deviation of recon_cosine above thr_top."""
        return np.maximum(0.0, recon - self.thr_top) / self.std_recon

    def _combined(self, seq: np.ndarray, recon: np.ndarray) -> np.ndarray:
        """Element-wise max of the two normalised deviations (OR-logic)."""
        return np.maximum(self._dev_seq(seq), self._dev_recon(recon))

    def score(self, scores_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Return per-sample anomaly scores for a single split.

        Returns dict with keys:
            'seq_score'    – deviation from bottom threshold
            'recon_score'  – deviation from top threshold
            'combined'     – OR-combined score (used for AUROC)
        """
        seq   = _extract(scores_dict, "seq_cosine")
        recon = _extract(scores_dict, "recon_cosine")
        return {
            "seq_score":   self._dev_seq(seq),
            "recon_score": self._dev_recon(recon),
            "combined":    self._combined(seq, recon),
        }


# ══════════════════════════════════════════════════════════════════════════════
# _compute_metrics – internal workhorse
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(
    detector: ThresholdDetector,
    origins_combined: np.ndarray,
    scores_dict: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-fraud AUROC, OOD rates, and score statistics.

    Parameters
    ----------
    detector        : fitted ThresholdDetector
    origins_combined: combined anomaly scores for the test-origins split
    scores_dict     : mapping  name -> {seq_cosine, recon_cosine, ...}
                      'origins' key is skipped automatically

    Returns
    -------
    dict  name -> {
        'n_samples', 'auroc',
        'ood_p95', 'ood_p99',
        'mean_seq', 'mean_recon', 'mean_combined',
        'std_seq', 'std_recon', 'std_combined',
        'roc_curve': {'fpr', 'tpr', 'thresholds'},
    }
    """
    metrics: Dict[str, Dict[str, Any]] = {}
    p95 = detector.p95_combined
    p99 = detector.p99_combined

    for name, sdict in scores_dict.items():
        if name == "origins":
            continue

        scored = detector.score(sdict)
        comb   = scored["combined"]
        seq_s  = scored["seq_score"]
        rec_s  = scored["recon_score"]

        # AUROC: origins=0, fraud=1; higher combined → more anomalous
        y_true  = np.concatenate([np.zeros(len(origins_combined)),
                                   np.ones(len(comb))])
        y_score = np.concatenate([origins_combined, comb])
        auroc_v = _safe_auroc(y_true, y_score)

        # ROC curve
        fpr, tpr, thr_roc = _safe_roc_curve(y_true, y_score)

        metrics[name] = {
            "n_samples":     int(len(comb)),
            "auroc":         float(auroc_v),
            "ood_p95":       float((comb > p95).mean()),
            "ood_p99":       float((comb > p99).mean()),
            "mean_seq":      float(seq_s.mean()),
            "mean_recon":    float(rec_s.mean()),
            "mean_combined": float(comb.mean()),
            "std_seq":       float(seq_s.std()),
            "std_recon":     float(rec_s.std()),
            "std_combined":  float(comb.std()),
            "roc_curve": {
                "fpr":        _to_list(fpr),
                "tpr":        _to_list(tpr),
                "thresholds": _to_list(thr_roc),
            },
        }
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# compute_threshold_auroc  –  single-level convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════

def compute_threshold_auroc(
    val_scores: Dict[str, np.ndarray],
    test_scores_map: Dict[str, Dict[str, np.ndarray]],
    q_bottom: float = 0.05,
    q_high: float = 0.95,
    level_name: str = "sequence",
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Fit ThresholdDetector on val, evaluate on test_scores_map.

    Parameters
    ----------
    val_scores      : dict  {signal: 1-D array}  (validation, in-distribution)
    test_scores_map : dict  name -> {signal: array}
                      Must contain an 'origins' key.
    q_bottom / q_high : quantile thresholds (see ThresholdDetector)
    level_name      : 'sequence' or 'video' – used in filenames / titles
    out_dir         : if given, writes threshold_results_{level_name}.json

    Returns
    -------
    dict with keys:
        'detector'   : ThresholdDetector object
        'thresholds' : {thr_bot, thr_top, std_seq, std_recon,
                        p95_combined, p99_combined}
        'metrics'    : name -> {auroc, ood_p95, ood_p99, …}  (individual level)
        'auroc'      : name -> float   (shortcut)
        'ood_rates'  : name -> {p95, p99, n}
    """
    assert "origins" in test_scores_map, (
        "test_scores_map must contain an 'origins' key."
    )
    assert "seq_cosine"   in val_scores, "val_scores must contain 'seq_cosine'."
    assert "recon_cosine" in val_scores, "val_scores must contain 'recon_cosine'."

    detector = ThresholdDetector(q_bottom=q_bottom, q_high=q_high).fit(val_scores)

    # Score test-origins (reference for AUROC)
    origins_combined = detector.score(test_scores_map["origins"])["combined"]

    metrics = _compute_metrics(detector, origins_combined, test_scores_map)

    results = {
        "detector":   detector,
        "thresholds": {
            "thr_bot":       detector.thr_bot,
            "thr_top":       detector.thr_top,
            "std_seq":       detector.std_seq,
            "std_recon":     detector.std_recon,
            "p95_combined":  detector.p95_combined,
            "p99_combined":  detector.p99_combined,
            "q_bottom":      detector.q_bottom,
            "q_high":        detector.q_high,
        },
        "metrics":   metrics,
        "auroc":     {n: m["auroc"]   for n, m in metrics.items()},
        "ood_rates": {
            n: {"p95": m["ood_p95"], "p99": m["ood_p99"], "n": m["n_samples"]}
            for n, m in metrics.items()
        },
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        serial = {
            "thresholds": results["thresholds"],
            "auroc":      results["auroc"],
            "ood_rates":  results["ood_rates"],
            "metrics":    {
                n: {k: v for k, v in m.items() if k != "roc_curve"}
                for n, m in metrics.items()
            },
        }
        path = out_dir / f"threshold_results_{level_name}.json"
        with open(path, "w") as fp:
            json.dump(serial, fp, indent=2, default=float)
        print(f"  ✓ Saved threshold detector summary → {path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# compute_stats_all_levels  –  mirrors gmm.py / ocsvm.py pattern
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats_all_levels(
    all_scores: Dict[str, Dict[str, np.ndarray]],
    th_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute ThresholdDetector statistics at three aggregation levels.

    Uses the already-fitted detector stored in th_results['detector']
    and the already-scored test-origins combined vector.

    Parameters
    ----------
    all_scores  : the full test scores map (individual variants),
                  including 'origins'.
    th_results  : return value of compute_threshold_auroc.

    Returns
    -------
    {
        'individual': {name: metrics_dict},   # raw _ID / _passport splits
        'merged':     {name: metrics_dict},   # _ID + _passport merged
        'grouped':    {name: metrics_dict},   # attack categories
        'composition': {group: {fraud: n}},   # sample counts
        'thresholds': …,
    }

    Each metrics_dict has the same keys as _compute_metrics output.
    """
    detector: ThresholdDetector = th_results["detector"]
    origins_combined = detector.score(all_scores["origins"])["combined"]

    # ── three aggregation levels ──────────────────────────────────────────────
    merged_scores  = merge_id_passport_variants(all_scores)
    grouped_scores = aggregate_to_groups(merged_scores)

    individual = _compute_metrics(detector, origins_combined, all_scores)
    merged     = _compute_metrics(detector, origins_combined, merged_scores)
    grouped    = _compute_metrics(detector, origins_combined, grouped_scores)

    return {
        "individual":   individual,
        "merged":       merged,
        "grouped":      grouped,
        "composition":  get_group_composition(merged_scores),
        "thresholds":   th_results["thresholds"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# print helpers
# ══════════════════════════════════════════════════════════════════════════════

_COL_W = 30   # fraud-name column width
_TABLE_W = 82  # total table width


def _print_table(
    title: str,
    data: Dict[str, Dict[str, Any]],
    level_name: str,
) -> None:
    """
    Print a box-drawn table of AUROC / OOD rates for a single aggregation level.

    Parameters
    ----------
    title      : header label
    data       : name -> metrics dict (output of _compute_metrics)
    level_name : 'sequence' or 'video' – shown in header
    """
    W = _TABLE_W
    w = _COL_W
    print(f"\n┌─ {title} ({level_name}) {'─' * max(0, W - len(title) - len(level_name) - 8)}┐")
    print(f"│ {'Fraud':<{w}s} │ {'N':>6s} │ {'AUROC':>7s} │ "
          f"{'OOD@p95':>8s} │ {'OOD@p99':>8s} │")
    print(f"├{'─' * (w+2)}┼{'─' * 8}┼{'─' * 9}┼{'─' * 10}┼{'─' * 10}┤")
    for name, m in sorted(data.items(), key=lambda x: -x[1]["auroc"]):
        auroc_s = f"{m['auroc']:.3f}" if not np.isnan(m["auroc"]) else "  NaN "
        print(f"│ {name:<{w}s} │ {m['n_samples']:>6d} │ {auroc_s:>7s} │ "
              f"{m['ood_p95']:>7.1%} │ {m['ood_p99']:>7.1%} │")
    print(f"└{'─' * (W)}┘")


def print_stats_summary(
    stats: Dict[str, Any],
    level_name: str = "sequence",
) -> None:
    """
    Pretty-print statistics at all three aggregation levels for one
    evaluation tier (sequence or video).

    Parameters
    ----------
    stats      : return value of compute_stats_all_levels
    level_name : 'sequence' or 'video'
    """
    W = _TABLE_W
    print(f"\n{'═' * W}")
    print(f"  THRESHOLD DETECTOR STATISTICS SUMMARY  ({level_name} level)")
    print(f"{'═' * W}")

    # ── Detector thresholds ──────────────────────────────────────────────────
    th = stats["thresholds"]
    print(f"  thr_bot (seq_cosine  < …)  = {th['thr_bot']:+.6f}   "
          f"[q_bottom={th['q_bottom']:.0%}]")
    print(f"  thr_top (recon_cosine > …) = {th['thr_top']:+.6f}   "
          f"[q_high  ={th['q_high']:.0%}]")
    print(f"  val combined p95={th['p95_combined']:.4f}  "
          f"p99={th['p99_combined']:.4f}")

    # ── Three aggregation tables ─────────────────────────────────────────────
    _print_table("GROUP LEVEL",      stats["grouped"],    level_name)
    _print_table("FRAUD TYPE LEVEL", stats["merged"],     level_name)
    _print_table("INDIVIDUAL LEVEL", stats["individual"], level_name)

    # ── Group composition ────────────────────────────────────────────────────
    print(f"\n┌─ GROUP COMPOSITION {'─' * (W - 22)}┐")
    for group_name, frauds in stats["composition"].items():
        total = sum(frauds.values())
        print(f"│  {group_name}  (n={total})")
        for fraud, n in frauds.items():
            pct = n / total * 100
            print(f"│    ├─ {fraud}: {n}  ({pct:.1f}%)")
    print(f"└{'─' * W}┘")


def print_stats_sequence_level(
    stats_seq: Dict[str, Any],
) -> None:
    """Print the sequence-level threshold-detector statistics summary."""
    print_stats_summary(stats_seq, level_name="sequence")


def print_stats_video_level(
    stats_vid: Dict[str, Any],
) -> None:
    """Print the video-level threshold-detector statistics summary."""
    print_stats_summary(stats_vid, level_name="video")


def print_stats_all_levels(
    stats_seq: Dict[str, Any],
    stats_vid: Dict[str, Any],
) -> None:
    """
    Print both sequence-level and video-level threshold-detector statistics.

    Parameters
    ----------
    stats_seq : return value of compute_stats_all_levels (sequence tier)
    stats_vid : return value of compute_stats_all_levels (video tier)
    """
    print_stats_sequence_level(stats_seq)
    print_stats_video_level(stats_vid)
