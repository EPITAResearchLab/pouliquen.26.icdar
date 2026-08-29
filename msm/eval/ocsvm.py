from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
from pathlib import Path
from .thresholds import auroc
from typing import Dict, Any
from .grouping import (
    merge_id_passport_variants,
    aggregate_to_groups,
    get_group_composition,
    FRAUD_GROUPS,
)


# ---------------------------------------------------------------------------
# Fitting & scoring
# ---------------------------------------------------------------------------

def fit_ocsvm_and_score(
    id_scores: dict[str, np.ndarray],
    test_scores_map: dict[str, dict[str, np.ndarray]],
    val_scores: dict[str, np.ndarray] | None = None,
    level_name: str = "sequence",
    out_dir: str | Path | None = None,
    nu: float = 0.05,
    kernel: str = "rbf",
    gamma: str = 0.1,  #"scale",
):
    """Fit OneClassSVM on ID (origins) 2D points and score all test sets.

    Parameters
    ----------
    id_scores : dict with keys 'recon_cosine' and 'seq_cosine' for origins
        (training/in-distribution data used to fit the model).
    test_scores_map : mapping name -> dict with same keys
        (includes 'origins' validation set and all fraud sets).
    val_scores : optional validation origins used for threshold computation.
        If None, id_scores is used for thresholds.
    level_name : 'sequence' or 'video', used in titles and filenames.
    out_dir : directory to save JSON summary (optional).
    nu : upper bound on the fraction of margin errors / support vectors.
        Effectively controls the sensitivity of the boundary (0 < nu <= 1).
        Lower values = tighter boundary = more OOD detections.
    kernel : kernel type for OneClassSVM ('rbf', 'poly', 'linear', etc.).
    gamma : kernel coefficient — 'scale', 'auto', or a float.

    Returns
    -------
    results dict with:
      'model'      : fitted OneClassSVM object
      'scaler'     : fitted StandardScaler
      'model_type' : 'ocsvm'
      'nu'         : nu value used
      'kernel'     : kernel used
      'scores'     : mapping name -> {'decision': array, 'anom': array}
      'thresholds' : p95, p99 on anomaly scores (from val_scores or id_scores)
      'auroc'      : mapping fraud name -> AUROC float
      'ood_rates'  : mapping fraud name -> {'p95': rate, 'p99': rate, 'n': count}
    """

    # ------------------------------------------------------------------
    # 1. Prepare feature arrays (N, 2)
    # ------------------------------------------------------------------
    def _to_xy(d):
        x = d["recon_cosine"]
        y = d["seq_cosine"]
        return np.vstack([x, y]).T

    origins_xy = _to_xy(id_scores)
    scaler = StandardScaler().fit(origins_xy)
    origins_xy_s = scaler.transform(origins_xy)

    # ------------------------------------------------------------------
    # 2. Fit OneClassSVM on ID data
    # ------------------------------------------------------------------
    print(f"[ocsvm] Fitting OneClassSVM  nu={nu}  kernel={kernel}  gamma={gamma}")
    model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    model.fit(origins_xy_s)
    print("[ocsvm] Fitting complete.")

    # ------------------------------------------------------------------
    # 3. Score all test sets
    #    decision_function: positive = inlier, negative = outlier.
    #    We negate so that a *higher* anom score means *more anomalous*.
    # ------------------------------------------------------------------
    results = {"model_type": "ocsvm", "nu": nu, "kernel": kernel}
    scores_out = {}
    for name, sdict in test_scores_map.items():
        xy = _to_xy(sdict)
        xy_s = scaler.transform(xy)
        decision = model.decision_function(xy_s)
        anom = -decision
        scores_out[name] = {"decision": decision, "anom": anom}
    results["scores"] = scores_out

    # ------------------------------------------------------------------
    # 4. Compute thresholds from validation (or ID) set
    # ------------------------------------------------------------------
    ref = val_scores if val_scores is not None else id_scores
    ref_xy = _to_xy(ref)
    ref_xy_s = scaler.transform(ref_xy)
    ref_decision = model.decision_function(ref_xy_s)
    ref_anom = -ref_decision

    p95 = float(np.percentile(ref_anom, 95))
    p99 = float(np.percentile(ref_anom, 99))
    results["thresholds"] = {
        "p95": p95,
        "p99": p99,
        "ref_mean": float(ref_anom.mean()),
        "val_ood_rate": float((ref_anom > 0.001).mean())
    }

    # ------------------------------------------------------------------
    # 5. Compute AUROC and OOD rates per fraud set
    # ------------------------------------------------------------------
    origins_anom = scores_out.get("origins", {"anom": ref_anom})["anom"]
    aurocs = {}
    ood_rates = {}
    for name, v in scores_out.items():
        if name == "origins":
            continue
        anom_vals = v["anom"]
        # origins = label 0 (in-distribution), fraud = label 1 (OOD)
        auroc_val = auroc(origins_anom, anom_vals)
        aurocs[name] = float(auroc_val)
        ood_rates[name] = {
            "p95": float((anom_vals > p95).mean()),
            "p99": float((anom_vals > p99).mean()),
            "n": len(anom_vals),
        }
    results["auroc"] = aurocs
    results["ood_rates"] = ood_rates

    # ------------------------------------------------------------------
    # 6. Optional: save JSON summary
    # ------------------------------------------------------------------
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(out_dir) / f"ocsvm_results_{level_name}.json"
        serial = {
            "model_type": "ocsvm",
            "nu": float(nu),
            "kernel": kernel,
            "thresholds": results["thresholds"],
            "auroc": results["auroc"],
            "ood_rates": results["ood_rates"],
        }
        with open(out_path, "w") as f:
            json.dump(serial, f, indent=2)
        print(f"Saved OneClassSVM summary → {out_path}")

    return {"model": model, "scaler": scaler, **results}


# ---------------------------------------------------------------------------
# Multi-level statistics
# ---------------------------------------------------------------------------

def compute_stats_all_levels(
    all_scores: Dict[str, Dict[str, np.ndarray]],
    ocsvm_results: Dict[str, Any],
    val_thresholds: Dict[str, Dict],
    auroc_fn,
) -> Dict[str, Any]:
    """Compute statistics at all aggregation levels.

    Parameters
    ----------
    all_scores : mapping name -> dict of score arrays for every set.
    ocsvm_results : return value of fit_ocsvm_and_score.
    val_thresholds : per-level threshold dicts (passed through for reference).
    auroc_fn : callable matching signature auroc(id_anom, ood_anom) -> float.

    Returns
    -------
    {
        'individual': {...},   # per _ID / _passport variant
        'merged':     {...},   # per base fraud type
        'grouped':    {...},   # per attack category
        'composition':{...},  # sample counts
    }
    """
    # Resolve origins anomaly scores for AUROC baseline
    origins_scores = all_scores.get("origins", {})
    origins_anom = ocsvm_results["scores"].get("origins", {}).get("anom")

    if origins_anom is None:
        # Fallback: score origins on-the-fly using the fitted model
        model = ocsvm_results["model"]
        scaler = ocsvm_results["scaler"]
        xy = np.vstack([
            origins_scores["recon_cosine"],
            origins_scores["seq_cosine"],
        ]).T
        origins_anom = -model.decision_function(scaler.transform(xy))

    p95 = ocsvm_results["thresholds"]["p95"]
    p99 = ocsvm_results["thresholds"]["p99"]

    def _compute_metrics(scores_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict:
        """Compute AUROC and OOD rates for each named entry."""
        model = ocsvm_results["model"]
        scaler = ocsvm_results["scaler"]

        metrics = {}
        for name, signals in scores_dict.items():
            if name == "origins":
                continue

            xy = np.vstack([signals["recon_cosine"], signals["seq_cosine"]]).T
            xy_s = scaler.transform(xy)
            anom = -model.decision_function(xy_s)

            auroc_val = auroc_fn(origins_anom, anom)
            metrics[name] = {
                "n_samples": len(anom),
                "auroc": float(auroc_val),
                "ood_p95": float((anom > p95).mean()),
                "ood_p99": float((anom > p99).mean()),
                "anom_mean": float(anom.mean()),
                "anom_std": float(anom.std()),
                "anom_median": float(np.median(anom)),
            }

        return metrics

    # Compute at each aggregation level
    merged_scores = merge_id_passport_variants(all_scores)
    grouped_scores = aggregate_to_groups(merged_scores)

    return {
        "individual": _compute_metrics(all_scores),
        "merged": _compute_metrics(merged_scores),
        "grouped": _compute_metrics(grouped_scores),
        "composition": get_group_composition(merged_scores),
    }


# ---------------------------------------------------------------------------
# Pretty-print summary  (unchanged from gmm.py)
# ---------------------------------------------------------------------------

def print_stats_summary(stats: Dict[str, Any], level_name: str = "sequence"):
    """Pretty-print statistics at all aggregation levels."""

    print(f"\n{'═' * 80}")
    print(f"  STATISTICS SUMMARY ({level_name} level)")
    print(f"{'═' * 80}")

    # ── Grouped level ──────────────────────────────────────────────────────
    print(f"\n┌─ GROUP LEVEL {'─' * 63}┐")
    print(f"│ {'Group':<30s} │ {'N':>6s} │ {'AUROC':>7s} │ {'OOD@p95':>8s} │ {'OOD@p99':>8s} │")
    print(f"├{'─' * 32}┼{'─' * 8}┼{'─' * 9}┼{'─' * 10}┼{'─' * 10}┤")

    for group_name in FRAUD_GROUPS.keys():
        if group_name not in stats["grouped"]:
            continue
        m = stats["grouped"][group_name]
        print(
            f"│ {group_name:<30s} │ {m['n_samples']:>6d} │ {m['auroc']:>7.3f} │ "
            f"{m['ood_p95']:>7.1%} │ {m['ood_p99']:>7.1%} │"
        )
    print(f"└{'─' * 78}┘")

    # ── Merged level ───────────────────────────────────────────────────────
    print(f"\n┌─ FRAUD TYPE LEVEL (merged _ID + _passport) {'─' * 33}┐")
    print(f"│ {'Fraud Type':<30s} │ {'N':>6s} │ {'AUROC':>7s} │ {'OOD@p95':>8s} │ {'OOD@p99':>8s} │")
    print(f"├{'─' * 32}┼{'─' * 8}┼{'─' * 9}┼{'─' * 10}┼{'─' * 10}┤")

    for fraud_name, m in sorted(stats["merged"].items(), key=lambda x: -x[1]["auroc"]):
        print(
            f"│ {fraud_name:<30s} │ {m['n_samples']:>6d} │ {m['auroc']:>7.3f} │ "
            f"{m['ood_p95']:>7.1%} │ {m['ood_p99']:>7.1%} │"
        )
    print(f"└{'─' * 78}┘")

    # ── Composition ────────────────────────────────────────────────────────
    print(f"\n┌─ GROUP COMPOSITION {'─' * 58}┐")
    for group_name, frauds in stats["composition"].items():
        total = sum(frauds.values())
        print(f"│ {group_name} (n={total})")
        for fraud, n in frauds.items():
            pct = n / total * 100
            print(f"│   ├─ {fraud}: {n} ({pct:.1f}%)")
    print(f"└{'─' * 78}┘")
