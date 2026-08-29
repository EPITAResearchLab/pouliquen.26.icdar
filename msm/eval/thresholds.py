import numpy as np


def compute_reference_thresholds(
    scores: dict[str, np.ndarray],
    percentiles: tuple[float, ...] = (1, 5, 95, 99),
) -> dict[str, dict]:
    ref = {}
    for key, vals in scores.items():
        ref[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "percentiles": {
                f"p{int(p)}": float(np.percentile(vals, p))
                for p in percentiles
            },
        }
    return ref


# ──────────────────────────────────────────────────────────────
# OOD evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_ood(
    id_scores: dict[str, np.ndarray],
    val_thresholds: dict[str, dict],
    test_scores: dict[str, np.ndarray],
    name: str,
) -> dict:
    """Evaluate a single fraud type against the in-distribution reference.

    Parameters
    ----------
    id_scores : per-signal arrays from the **test origins** set (used for AUROC).
    val_thresholds : threshold statistics from the **validation** set (used for OOD rates).
    test_scores : per-signal arrays for the fraud type being evaluated.
    name : human-readable name of the fraud type.

    Returns
    -------
    dict with n_samples, and per-signal stats including AUROC (test-only)
    and OOD rates (thresholded by validation percentiles).
    """
    result: dict = {"name": name, "n_samples": len(next(iter(test_scores.values())))}
    signals = {}

    for key in test_scores:
        t_vals = test_scores[key]
        id_vals = id_scores[key]
        th = val_thresholds[key]

        p95 = th["percentiles"]["p95"]
        p99 = th["percentiles"]["p99"]

        signals[key] = {
            "mean": float(t_vals.mean()),
            "std": float(t_vals.std()),
            "min": float(t_vals.min()),
            "max": float(t_vals.max()),
            "ood_rate_p95": float((t_vals > p95).mean()),
            "ood_rate_p99": float((t_vals > p99).mean()),
            "auroc": float(auroc(id_vals, t_vals)),
        }

    result["signals"] = signals
    return result


def auroc(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """Compute AUROC without sklearn.

    Labels:  in-distribution → 0,  OOD → 1.
    Higher score = more anomalous.
    """
    labels = np.concatenate([
        np.zeros(len(scores_id)),
        np.ones(len(scores_ood)),
    ])
    scores = np.concatenate([scores_id, scores_ood])

    order = np.argsort(-scores)
    labels_sorted = labels[order]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0

    for lab in labels_sorted:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        auc += (fp - prev_fp) * (tp + prev_tp) / 2
        prev_fp = fp
        prev_tp = tp

    return auc / (n_pos * n_neg)

