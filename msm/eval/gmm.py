from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
from pathlib import Path
from .thresholds import auroc
from sklearn.metrics import roc_auc_score, roc_curve

# Prepare arrays (N,2)
def _to_xy(d):
    x = d["recon_cosine"]
    # y = d["recon_mse"]
    y = d["seq_cosine"]
    return np.vstack([x, y]).T

def fit_gmm_and_score(
    id_scores: dict[str, np.ndarray],
    test_scores_map: dict[str, dict[str, np.ndarray]],
    val_scores: dict[str, np.ndarray] | None = None,
    level_name: str = "sequence",
    out_dir: str | Path | None = None,
    cov_type: str = "full",
    random_state: int = 42,
):
    """Fit GMM on ID (origins) 2D points and score all test sets.

    Parameters
    ----------
    id_scores: dict with keys 'recon_cosine' and 'recon_mse' for origins (training/test origins)
    test_scores_map: mapping name -> dict with same keys (includes 'origins' and frauds)
    val_scores: optional validation origins (for thresholds). If None, uses id_scores for thresholds.
    level_name: 'sequence' or 'video' used in titles/filenames
    out_dir: where to save JSON / plots (optional)
    cov_type: one of 'full','tied','diag','spherical'
    Returns
    -------
    results dict with:
      'gmm' : sklearn.GaussianMixture object (fitted)
      'scaler' : StandardScaler fitted
      'scores' : mapping name -> {'loglik': array, 'anom': array}
      'thresholds' : p95,p99 on anomaly (from val_scores or id_scores)
      'auroc' : mapping fraud -> AUROC
      'ood_rates' : mapping fraud -> {'p95': rate, 'p99': rate}
    """

    # origins_xy = _to_xy(id_scores)
    # scaler = StandardScaler().fit(origins_xy)
    # origins_xy_s = scaler.transform(origins_xy)

    val_xy  = _to_xy(val_scores)                          # (N_val, 2)
    scaler  = StandardScaler().fit(val_xy)
    val_xys = scaler.transform(val_xy)

    gmm = GaussianMixture(
        n_components=1,
        covariance_type=cov_type,
        reg_covar=1e-6,
        random_state=random_state,
        n_init=3,
    )
    gmm.fit(val_xys)

    val_anom    = -gmm.score_samples(val_xys)             # higher = more anomalous
    p95         = float(np.percentile(val_anom, 95))
    p99         = float(np.percentile(val_anom, 99))

    test_scores = {}
    for name, sdict in test_scores_map.items():
        xy     = _to_xy(sdict)
        xys    = scaler.transform(xy)
        loglik = gmm.score_samples(xys)                   # log p(x)
        anom   = -loglik                                   # anomaly score
        test_scores[name] = {"loglik": loglik, "anom": anom}

    
        origins_anom = test_scores["origins"]["anom"]         # ID test reference

    val_anom    = -gmm.score_samples(val_xys)             # higher = more anomalous
    p95         = float(np.percentile(val_anom, 95))
    p99         = float(np.percentile(val_anom, 99))

    # ── 4. Score ALL test splits ──────────────────────────────────────────────
    test_scores: Dict[str, Dict[str, np.ndarray]] = {}
    for name, sdict in test_scores_map.items():
        xy     = _to_xy(sdict)
        xys    = scaler.transform(xy)
        loglik = gmm.score_samples(xys)                   # log p(x)
        anom   = -loglik                                   # anomaly score
        test_scores[name] = {"loglik": loglik, "anom": anom}

    # ── 5. AUROC + ROC curves + OOD rates  (test set only) ───────────────────
    origins_anom = test_scores["origins"]["anom"]         # ID test reference

    aurocs:     Dict[str, float]          = {}
    roc_curves: Dict[str, Dict]           = {}
    ood_rates:  Dict[str, Dict[str, Any]] = {}

    for name, v in test_scores.items():
        fraud_anom = v["anom"]

        # OOD rates against val thresholds (applies to every split, incl. origins)
        ood_rates[name] = {
            "p95": float((fraud_anom > p95).mean()),
            "p99": float((fraud_anom > p99).mean()),
            "n":   int(len(fraud_anom)),
        }

        if name == "origins":
            continue  # skip AUROC for the reference split itself

        # Build binary labels: 0 = ID (origins), 1 = fraud
        y_true  = np.concatenate([np.zeros(len(origins_anom)),
                                   np.ones(len(fraud_anom))])
        y_score = np.concatenate([origins_anom, fraud_anom])

        aurocs[name] = roc_auc_score(y_true, y_score)

        fpr, tpr, thresh = roc_curve(y_true, y_score)
        roc_curves[name] = (
            {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
             "thresholds": thresh.tolist()}
            if fpr is not None else {}
        )

    # ── 6. Optional JSON export ───────────────────────────────────────────────
    results = {
        "gmm":               gmm,
        "scaler":            scaler,
        "val_threshold_p95": p95,
        "val_threshold_p99": p99,
        # keep legacy key names so the rest of the codebase keeps working
        "thresholds":        {"p95": p95, "p99": p99,
                              "ref_mean": float(val_anom.mean())},
        "test_scores":       test_scores,
        "scores":            test_scores,   # alias for plotting helpers
        "auroc":             aurocs,
        "roc_curves":        roc_curves,
        "ood_rates":         ood_rates,
        "gmm_n_components":  1,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── 0. Shared score pools ─────────────────────────────────────────
        # origins = test-split legit anomaly scores (negative class, label 0)
        # fraud keys = every key that appears in aurocs (positive class, label 1)
        origins_anom: np.ndarray = results["scores"]["origins"]["anom"]
        fraud_scores_roc: dict[str, np.ndarray] = {
            name: results["scores"][name]["anom"]
            for name in results["auroc"]          # fraud keys only
        }

        # ── 1. Existing GMM summary (unchanged) ───────────────────────────
        summary_path = out_dir / f"gmm_results_{level_name}.json"
        serial = {
            "gmm_n_components": 1,
            "covariance_type":  cov_type,
            "thresholds":       results["thresholds"],
            "auroc":            {k: (v if np.isfinite(v) else None)
                                 for k, v in aurocs.items()},
            "ood_rates":        ood_rates,
        }
        with open(summary_path, "w") as f:
            json.dump(serial, f, indent=2)
        print(f"Saved GMM summary       → {summary_path}")

        # ── 2. Raw scores — roc_data_{level}.npz + roc_data_{level}.json ────
        #    Mirrors pipeline_with_roc._save_roc_data_npz layout exactly:
        #      arr_origins        ← legit anomaly scores
        #      arr_<dataset>      ← fraud anomaly scores (hyphens → underscores)
        npz_arrays: dict[str, np.ndarray] = {"arr_origins": origins_anom}
        for ds, sc in fraud_scores_roc.items():
            safe_key = "arr_" + ds.replace("-", "_").replace(" ", "_")
            npz_arrays[safe_key] = sc

        roc_npz_path = out_dir / f"roc_data_{level_name}.npz"
        np.savez_compressed(roc_npz_path, **npz_arrays)
        print(f"Saved ROC raw scores    → {roc_npz_path}")

        roc_json: dict = {
            "pipeline":        "gmm",
            "score_name":      f"GMM anomaly · -log p(x) · {level_name}-level",
            "score_direction": "higher_is_more_anomalous",
            "origins":  origins_anom.tolist(),
            "datasets": {ds: sc.tolist() for ds, sc in fraud_scores_roc.items()},
        }
        roc_json_path = out_dir / f"roc_data_{level_name}.json"
        with open(roc_json_path, "w") as f:
            json.dump(roc_json, f)
        print(f"Saved ROC raw json      → {roc_json_path}")

        # ── 3. Precomputed ROC curves — roc_curves_{level}.npz + .json ───────
        #    Mirrors pipeline_with_roc._save_roc_curves layout exactly.
        #    Convention:
        #      y_true  = [0]*n_origins + [1]*n_fraud
        #      y_score = concat(origins_anom, fraud_anom)
        #    NPZ keys per dataset (hyphens → underscores):
        #      fpr_<ds>, tpr_<ds>, thr_<ds>
        #    JSON: manifest with pipeline metadata + per-dataset curve arrays.
        curves_npz_arrays:    dict[str, np.ndarray] = {}
        curves_json_datasets: dict[str, dict]       = {}

        for ds, fraud_scores in fraud_scores_roc.items():
            y_true  = np.array([0] * len(origins_anom) + [1] * len(fraud_scores))
            y_score = np.concatenate([origins_anom, fraud_scores])
            fpr, tpr, thr = roc_curve(y_true, y_score)

            safe = ds.replace("-", "_").replace(" ", "_")
            curves_npz_arrays[f"fpr_{safe}"] = fpr
            curves_npz_arrays[f"tpr_{safe}"] = tpr
            curves_npz_arrays[f"thr_{safe}"] = thr

            curves_json_datasets[ds] = {
                "auc":        float(aurocs.get(ds, float("nan"))),
                "n_fraud":    int(len(fraud_scores)),
                "n_origins":  int(len(origins_anom)),
                "fpr":        fpr.tolist(),
                "tpr":        tpr.tolist(),
                "thresholds": thr.tolist(),
            }

        curves_npz_path = out_dir / f"roc_curves_{level_name}.npz"
        np.savez_compressed(curves_npz_path, **curves_npz_arrays)
        print(f"Saved ROC curves (npz)  → {curves_npz_path}")

        curves_manifest = {
            "pipeline":        "gmm",
            "score_name":      f"GMM anomaly · -log p(x) · {level_name}-level",
            "score_direction": "higher_is_more_anomalous",
            "datasets":        curves_json_datasets,
        }
        curves_json_path = out_dir / f"roc_curves_{level_name}.json"
        with open(curves_json_path, "w") as f:
            json.dump(curves_manifest, f, indent=2)
        print(f"Saved ROC curves (json) → {curves_json_path}")

    return results

# evaluation/stats.py
"""Statistics computation at multiple aggregation levels."""

import numpy as np
from typing import Dict, Any
from .grouping import (
    merge_id_passport_variants,
    aggregate_to_groups,
    get_group_composition,
    FRAUD_GROUPS,
)

def compute_stats_all_levels(
    all_scores:     Dict[str, Dict[str, np.ndarray]],
    gmm_results:    Dict[str, Any],
    val_thresholds: Dict[str, Dict],   # kept for API compat
    auroc_fn,
) -> Dict[str, Any]:
    """
    Compute AUROC + OOD rates at individual / merged / grouped levels.
    Uses the test-split 'origins' from gmm_results['test_scores'] as reference.
    Thresholds come from gmm_results['thresholds'] (derived from validation).
    """
    gmm    = gmm_results["gmm"]
    scaler = gmm_results["scaler"]
    p95    = gmm_results["thresholds"]["p95"]
    p99    = gmm_results["thresholds"]["p99"]

    # Reference anomaly scores — test-split origins (NOT val)
    origins_anom = gmm_results["test_scores"]["origins"]["anom"]

    def _score(sdict: Dict[str, np.ndarray]) -> np.ndarray:
        """Anomaly score for an arbitrary split dict."""
        xy  = _to_xy(sdict)
        xys = scaler.transform(xy)
        return -gmm.score_samples(xys)

    def _compute_metrics(scores_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict:
        metrics = {}
        for name, signals in scores_dict.items():
            if name == "origins":
                continue
            anom = _score(signals)

            y_true  = np.concatenate([np.zeros(len(origins_anom)),
                                       np.ones(len(anom))])
            y_score = np.concatenate([origins_anom, anom])

            fpr, tpr, thresolds = roc_curve(y_true=y_true,
                      y_score=y_score)
            roc_dict = {
                "fpr":        list(fpr),
                "tpr":        list(tpr),
                "thresholds": list(thresolds),
            }

            metrics[name] = {
                "n_samples":  int(len(anom)),
                "auroc":      roc_auc_score(y_true, y_score),
                "ood_p95":    float((anom > p95).mean()),
                "ood_p99":    float((anom > p99).mean()),
                "anom_mean":  float(anom.mean()),
                "anom_std":   float(anom.std()),
                "anom_median":float(np.median(anom)),
                "roc_curve":   roc_dict
            }
        return metrics

    merged_scores  = merge_id_passport_variants(all_scores)
    grouped_scores = aggregate_to_groups(merged_scores)

    return {
        "individual":  _compute_metrics(all_scores),
        "merged":      _compute_metrics(merged_scores),
        "grouped":     _compute_metrics(grouped_scores),
        "composition": get_group_composition(merged_scores),
    }


def print_stats_summary(stats: Dict[str, Any], level_name: str = "sequence"):
    """Pretty-print statistics at all levels."""
    
    print(f"\n{'═' * 80}")
    print(f"  STATISTICS SUMMARY ({level_name} level)")
    print(f"{'═' * 80}")
    
    # ── Grouped level ─────────────────────────────────────────
    print(f"\n┌─ GROUP LEVEL {'─' * 63}┐")
    print(f"│ {'Group':<30s} │ {'N':>6s} │ {'AUROC':>7s} │ {'OOD@p95':>8s} │ {'OOD@p99':>8s} │")
    print(f"├{'─' * 32}┼{'─' * 8}┼{'─' * 9}┼{'─' * 10}┼{'─' * 10}┤")
    
    for group_name in FRAUD_GROUPS.keys():
        if group_name not in stats["grouped"]:
            continue
        m = stats["grouped"][group_name]
        print(f"│ {group_name:<30s} │ {m['n_samples']:>6d} │ {m['auroc']:>7.3f} │ "
              f"{m['ood_p95']:>7.1%} │ {m['ood_p99']:>7.1%} │")
    print(f"└{'─' * 78}┘")
    
    # ── Merged level (base fraud types) ───────────────────────
    print(f"\n┌─ FRAUD TYPE LEVEL (merged _ID + _passport) {'─' * 33}┐")
    print(f"│ {'Fraud Type':<30s} │ {'N':>6s} │ {'AUROC':>7s} │ {'OOD@p95':>8s} │ {'OOD@p99':>8s} │")
    print(f"├{'─' * 32}┼{'─' * 8}┼{'─' * 9}┼{'─' * 10}┼{'─' * 10}┤")
    
    for fraud_name, m in sorted(stats["merged"].items(), key=lambda x: -x[1]["auroc"]):
        print(f"│ {fraud_name:<30s} │ {m['n_samples']:>6d} │ {m['auroc']:>7.3f} │ "
              f"{m['ood_p95']:>7.1%} │ {m['ood_p99']:>7.1%} │")
    print(f"└{'─' * 78}┘")
    
    # ── Composition ───────────────────────────────────────────
    print(f"\n┌─ GROUP COMPOSITION {'─' * 58}┐")
    for group_name, frauds in stats["composition"].items():
        total = sum(frauds.values())
        print(f"│ {group_name} (n={total})")
        for fraud, n in frauds.items():
            pct = n / total * 100
            print(f"│   ├─ {fraud}: {n} ({pct:.1f}%)")
    print(f"└{'─' * 78}┘")
