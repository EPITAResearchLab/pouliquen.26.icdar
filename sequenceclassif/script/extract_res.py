import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

groups = {
    "static_midvholo": [
        "copy_without_holo",
        "pseudo_holo_copy",
        "photo_holo_copy",
    ],
    "static_midvdynattack": [
        "no_holo",
        "laser",
        "plastified_led",
        "plastified_lowreflect",
        "plastified_noholo",
    ],
    "dynamic_midvdynattack": [
        "holo_star_world",
        "holo_completemask",
        "leaf_holo",
        "plain_holo",
        "double_sticker",
    ],
    "swap_midvdynattack": [
        "swap",
        "swap_three",
    ],
}

def find_group(g):
    for gr, v in groups.items():
        if g in v:
            return gr
    return g


def average_results_per_video(results_full):
    """
    Average results per video instead of per sequence
    
    Args:
        results_full: Dict with structure {category: [[score, video_idx], ...]}
    
    Returns:
        results_avg: Dict with averaged scores per video {category: [avg_scores]}
    """
    results_avg = {}
    
    for category, score_video_pairs in results_full.items():
        # Group scores by video_idx
        video_scores = defaultdict(list)
        
        for score, video_idx in score_video_pairs:
            video_scores[video_idx].append(score)

        print(category, len(video_scores))
        
        # Calculate average score per video
        avg_scores = [np.mean(scores) for scores in video_scores.values()]
        results_avg[category] = avg_scores
        
        print(f"{category}: {len(score_video_pairs)} sequences -> {len(avg_scores)} videos")
    
    return results_avg

def get_metrics_from_values(
    acc_origins: np.array, acc_frauds: np.array,
) -> dict[str, float]:
    """Compute metrics from two lists.

    Returns:
        dict with metrics
    """
    # acc_frauds is what was predicted, 1 means that its a tp
    # calculating stats
    origin_fp = acc_origins
    frauds_tp = acc_frauds

    tp = sum(frauds_tp)
    fp = sum(origin_fp)
    fn = len(frauds_tp) - tp
    tn = len(origin_fp) - fp
    if len(acc_frauds) == 0:  # only origins
        return {
            "specificity": 1 - sum(acc_origins) / len(acc_origins),
            "len": len(acc_origins),
        }
    if len(acc_origins) == 0:  # only frauds
        return {"recall": sum(acc_frauds) / len(acc_frauds), "len": len(acc_frauds)}

    precision = 0
    recall = 0
    fscore = 0

    sumof = sum(acc_origins) + sum(acc_frauds)
    if sumof:
        precision = sum(acc_frauds) / sumof
        recall = sum(acc_frauds) / len(acc_frauds)
        if not precision or not recall:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)

    return {
        "fscore": fscore,
        "recall": recall,
        "precision": precision,
        "specificity": 1 - sum(acc_origins) / len(acc_origins),
        "fp": fp,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "len": len(acc_origins) + len(acc_frauds),
    }


def analyze_atthreshold(results_avg, threshold=0.5):
    threshold_stats = {}
    
    print(f"\n=== THRESHOLD ANALYSIS (threshold = {threshold}) ===")
    
    for category, scores in results_avg.items():
        above_threshold = sum(1 for score in scores if score > threshold)
        total_videos = len(scores)
        percentage = (above_threshold / total_videos) * 100 if total_videos > 0 else 0
        
        threshold_stats[category] = {
            'above_threshold': above_threshold,
            'total': total_videos,
            'percentage': percentage
        }
        
        if category == 'origins':  # Last category (origins)
            print(f"{category} (legitimate): {above_threshold}/{total_videos} videos above {threshold} ({percentage:.1f}% outliers)")
        else:  # Fraud categories
            print(f"{category} (fraud): {above_threshold}/{total_videos} videos above {threshold} ({percentage:.1f}%)")
    
    return threshold_stats, threshold


group = True
results_avgs_val = None
results_avgs_val_list = []
results_avgs_test = None
results_avgs_test_list = []

runname = "phd_f_final"

# Simple version with just averaged boxplots and threshold analysis
for k in range(5):
    with open(f"sequenceclassif/jsonres/k{k}-{runname}_origins_full_val.json") as f:

        results_full = json.load(f)
    results_full_group = {g: [] for g in (groups if group else results_full)}
    for name, v in results_full.items():
        video_scores = defaultdict(list)
        for score, video_idx in v:
            # print(score, video_idx)
            video_scores[video_idx].append(score)
        # print(f"{video_scores=}")
        avg_scores = [np.mean(scores) for scores in video_scores.values()]
        # results_avg[category] = avg_scores
        # print(f"{avg_scores=}")

        g = find_group(name) if group else name
        print(name)
        if g in results_full_group:
            # group by video before
            results_full_group[g].extend(avg_scores)
            # print("adding", name, "to")
        else:
            results_full_group[g] = avg_scores

    # results_avg = average_results_per_video(v)
    # results_full_group = results_full
    # results_avg = results_full_group

    # # Average per video
    results_avgs_val_list.append(results_full_group)
    if results_avgs_val is None:
        results_avgs_val = results_full_group.copy()
    else:
        results_avgs_val = {key: np.concatenate((value, results_full_group[key])) for key, value in results_avgs_val.items()}

    # with open(f"k{k}-finalfinal_origins_full.json") as f:
    with open(f"sequenceclassif/jsonres/k{k}-{runname}_origins_full_test.json") as f:

        results_full_val = json.load(f)
    results_full_val_group = {g: [] for g in (groups if group else results_full_val)}
    for name, v in results_full_val.items():
        video_scores = defaultdict(list)
        for score, video_idx in v:
            # print(score, video_idx)
            video_scores[video_idx].append(score)
        # print(f"{video_scores=}")
        avg_scores = [np.mean(scores) for scores in video_scores.values()]
        # results_avg[category] = avg_scores
        # print(f"{avg_scores=}")

        g = find_group(name) if group else name
        print(name)
        if g in results_full_val_group:
            # group by video before
            results_full_val_group[g].extend(avg_scores)
            # print("adding", name, "to")
        else:
            results_full_val_group[g] = avg_scores

    # results_avg = average_results_per_video(v)
    # results_full_val_group = results_full_val
    # results_avg = results_full_val_group

    # # Average per video
    results_avgs_test_list.append(results_full_val_group)
    if results_avgs_test is None:
        results_avgs_test = results_full_val_group.copy()
    else:
        results_avgs_test = {key: np.concatenate((value, results_full_val_group[key])) for key, value in results_avgs_test.items()}


values_frauds, values_origins = 1- results_avgs_val["static_midvholo"], 1-results_avgs_val["origins"]
metrics_m = {"fscore": -1}
th = 0

full = np.concatenate((values_frauds, values_origins, [0, 1, 1.1]))
for i in np.unique(full):
    # tp is a frauds predicted as fraud
    origin = values_origins < i
    frauds = values_frauds < i
    metrics = get_metrics_from_values(origin, frauds)
    if metrics["fscore"] > metrics_m["fscore"]:
        th = i
        metrics_m = metrics
th = 1 - th
print(metrics_m, th)

# Analyze threshold
threshold_stats, threshold = analyze_atthreshold(results_avgs_test, th)

# Convert to list format for boxplot
category_names = list(results_avgs_test.keys())
print(category_names)
results_list = [results_avgs_test[cat] for cat in category_names]

# Create the plot
plt.figure(figsize=(15, 10))
boxplot = sns.boxplot(data=results_list, palette=["r"] * (len(results_avgs_test)-1) + ["g"])

# Add threshold line
plt.axhline(y=threshold, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f"Threshold = {th:.2f}")

# Add text annotations showing counts above threshold
for i, category in enumerate(category_names):
    stats = threshold_stats[category]
    above_count = stats['above_threshold']
    total_count = stats['total']
    percentage = stats['percentage']
    
    # Position text above the boxplot

    y_pos = max(results_avgs_test[category]) + 0.02  # Slightly above max value
    print(y_pos)
    if category == 'origins':  # Last category (legitimate)
        text = f"Outliers: {above_count}/{total_count}\n({percentage:.1f}%)"
        color = 'red'
    else:  # Fraud categories
        text = f"Above: {above_count}/{total_count}\n({percentage:.1f}%)"
        color = 'darkred'
    
    plt.text(i, y_pos, text, ha='center', va='bottom', fontsize=10, 
            color=color, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.title(f"Model predictions per VIDEO (averaged) - ALL FOLDS")
plt.xlabel('Categories')
plt.ylabel('Average Prediction Score per Video')
plt.xticks(range(len(category_names)), category_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Adjust layout to make room for annotations
plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.savefig(f"phd_kall_video_avg_threshold.png", dpi=300, bbox_inches='tight')
plt.savefig(f"phd_video_avg_threshold.pdf", bbox_inches='tight')
plt.show()
plt.close()

print(f"\n{'=' * 50}")

def get_metrics_from_values(
    acc_origins: np.array, acc_frauds: np.array,
) -> dict[str, float]:
    """Compute metrics from two lists.

    Returns:
        dict with metrics
    """
    # acc_frauds is what was predicted, 1 means that its a tp
    # calculating stats
    origin_fp = acc_origins
    frauds_tp = acc_frauds

    tp = sum(frauds_tp)
    fp = sum(origin_fp)
    fn = len(frauds_tp) - tp
    tn = len(origin_fp) - fp
    if len(acc_frauds) == 0:  # only origins
        return {
            "specificity": 1 - sum(acc_origins) / len(acc_origins),
            "len": len(acc_origins),
        }
    if len(acc_origins) == 0:  # only frauds
        return {"recall": sum(acc_frauds) / len(acc_frauds), "len": len(acc_frauds)}

    precision = 0
    recall = 0
    fscore = 0

    sumof = sum(acc_origins) + sum(acc_frauds)
    if sumof:
        precision = sum(acc_frauds) / sumof
        recall = sum(acc_frauds) / len(acc_frauds)
        if not precision or not recall:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)

    return {
        "fscore": fscore,
        "recall": recall,
        "precision": precision,
        "specificity": 1 - sum(acc_origins) / len(acc_origins),
        "fp": fp,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "len": len(acc_origins) + len(acc_frauds),
    }





from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from pathlib import Path

# =============================================================================
# AUC + Calibration @99 / @95  (val → test)
# =============================================================================
# Calibration strategy
#   @99  => threshold on val where FPR ≤ 1 %  (specificity = 99 %)
#   @95  => threshold on val where FPR ≤ 5 %  (specificity = 95 %)
# =============================================================================


# ── Dataset / group definitions ───────────────────────────────────────────────

FRAUD_DATASETS = [
    "static_midvholo",
    "static_midvdynattack",
    "dynamic_midvdynattack",
    "swap_midvdynattack",
    "photo_replacement",
]

DATASET_WEIGHTS = {
    "static_midvholo":        60,
    "static_midvdynattack":  110,
    "dynamic_midvdynattack":  90,
    "swap_midvdynattack":     40,
    "photo_replacement":      20,
}

DATASET_TO_GROUP = {
    "static_midvholo":       "midvholo",
    "static_midvdynattack":  "midvdyn",
    "dynamic_midvdynattack": "midvdyn",
    "swap_midvdynattack":    "midvdyn",
    "photo_replacement":     "photorep",
}

# Operating points: label → target FPR on the val set
OPERATING_POINTS = {
    "@99": 0.01,   # accept 99 % of genuine users  =>  FPR = 1 %
    "@95": 0.05,   # accept 95 % of genuine users  =>  FPR = 5 %
}

ROC_OUT_DIR = "roc_folds"

def auroc_vs_origins(
    fraud_scores: np.ndarray,
    origin_scores: np.ndarray,
):
    """
    Exact binary AUROC: origins = 0, fraud = 1.
    Returns (auc_val, fpr, tpr, thresholds).
    """
    y_true  = np.array([0] * len(origin_scores) + [1] * len(fraud_scores))
    y_score = np.concatenate([origin_scores, fraud_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    auc_val = roc_auc_score(y_true, y_score)
    return auc_val, fpr, tpr, thresholds


def calibrate_threshold_at_fpr(
    origin_val_scores: np.ndarray,
    target_fpr: float,
) -> float:
    """
    Return the score threshold such that at most `target_fpr` fraction of
    validation origins are flagged (score ≥ threshold).

    Derivation:
        FPR = #{origins : score ≥ th} / N_origins  ≤ target_fpr
        ⟹ th = quantile(origin_scores, 1 - target_fpr)
    """
    return float(np.quantile(origin_val_scores, 1.0 - target_fpr))


def recall_at_threshold(scores: np.ndarray, threshold: float) -> float:
    """TPR: fraction of fraud scores ≥ threshold (detected as fraud)."""
    return float(np.mean(np.asarray(scores) >= threshold))


def fpr_at_threshold(origin_scores: np.ndarray, threshold: float) -> float:
    """Actual FPR on a given split (may differ from val calibration target)."""
    return float(np.mean(np.asarray(origin_scores) >= threshold))


def _empty_group() -> dict:
    return {"fpr": [], "tpr": [], "thresholds": [], "auc": []}


def save_fold_roc(
    fold: int,
    groups: dict,
    *,
    method: str = "gmm",
    score_direction: str = "higher_is_more_anomalous",
    out_dir: str = ROC_OUT_DIR,
) -> str:
    manifest = {
        "method":          method,
        "fold":            fold,
        "score_direction": score_direction,
        "groups":          groups,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"fold_{fold}_roc_curve.json"
    with open(out_path, "w") as fh:
        json.dump(manifest, fh)
    print(f"[ROC] fold {fold} → {out_path}")
    return str(out_path)


# ── Main loop ─────────────────────────────────────────────────────────────────

rows = []

for k in range(5):
    res_val  = results_avgs_val_list[k]
    res_test = results_avgs_test_list[k]

    origins_val  = np.array(res_val["origins"])
    origins_test = np.array(res_test["origins"])

    # ── 1. AUC on test ────────────────────────────────────────────────────────
    auc_per_ds: dict = {}
    roc_groups: dict = defaultdict(_empty_group)

    for ds in FRAUD_DATASETS:
        if ds not in res_test:
            continue
        auc_val, fpr, tpr, thresholds = auroc_vs_origins(
            np.array(res_test[ds]), origins_test
        )
        auc_per_ds[ds] = auc_val

        group = DATASET_TO_GROUP.get(ds, "other")
        roc_groups[group]["fpr"].append(fpr.tolist())
        roc_groups[group]["tpr"].append(tpr.tolist())
        roc_groups[group]["thresholds"].append(thresholds.tolist())
        roc_groups[group]["auc"].append(float(auc_val))

    save_fold_roc(fold=k, groups=dict(roc_groups), method="gmm", out_dir=ROC_OUT_DIR)

    total_w      = sum(DATASET_WEIGHTS[ds] for ds in auc_per_ds)
    auc_combined = (
        sum(DATASET_WEIGHTS[ds] * auc_per_ds[ds] for ds in auc_per_ds) / total_w
    )

    # ── 2. Calibrate thresholds on VAL (@99 and @95) ──────────────────────────
    thresholds_cal: dict = {}
    for tag, target_fpr in OPERATING_POINTS.items():
        th = calibrate_threshold_at_fpr(origins_val, target_fpr)
        thresholds_cal[tag] = th

    # ── 3. Evaluate calibrated thresholds on TEST ─────────────────────────────
    row = {
        "fold":         k,
        "auc_combined": auc_combined,
        **{f"auc_{ds}": v for ds, v in auc_per_ds.items()},
    }

    for tag, th in thresholds_cal.items():
        # Actual FPR on test (sanity-check — should be close to the target)
        row[f"fpr_test{tag}"]  = fpr_at_threshold(origins_test, th)
        row[f"threshold{tag}"] = th
        for ds in FRAUD_DATASETS:
            if ds not in res_test:
                continue
            row[f"recall_{ds}{tag}"] = recall_at_threshold(res_test[ds], th)

    rows.append(row)

    print(
        f"Fold {k} │ AUC_combined={auc_combined:.4f} "
        + "  ".join(
            f"│ th{tag}={thresholds_cal[tag]:.4f}  "
            f"fpr_test={row[f'fpr_test{tag}']:.3f}  "
            f"recall_midvholo={row[f'recall_static_midvholo{tag}']:.3f}"
            for tag in OPERATING_POINTS
        )
    )


# ── Summary DataFrame ─────────────────────────────────────────────────────────

df       = pd.DataFrame(rows)
num_cols = [c for c in df.columns if c != "fold"]

df_summary = pd.concat(
    [
        df,
        pd.DataFrame(
            [
                {
                    "fold": "mean±std",
                    **{
                        c: f"{df[c].mean()*100:.1f} ± {df[c].std()*100:.1f}"
                        for c in num_cols
                    },
                }
            ]
        ),
    ],
    ignore_index=True,
)


# ── Formatted display ─────────────────────────────────────────────────────────

SEP = "=" * 110

# AUC block
print(f"\n{SEP}")
print("AUC (test)")
auc_cols = ["fold", "auc_combined"] + [f"auc_{ds}" for ds in FRAUD_DATASETS if f"auc_{ds}" in df.columns]
print(df_summary[auc_cols].to_string(index=False))

# Per-operating-point raw tables
for tag in OPERATING_POINTS:
    print(f"\n{SEP}")
    print(f"Operating point {tag}  (threshold calibrated on val, evaluated on test)")
    op_cols = (
        ["fold", f"threshold{tag}", f"fpr_test{tag}"]
        + [f"recall_{ds}{tag}" for ds in FRAUD_DATASETS if f"recall_{ds}{tag}" in df.columns]
    )
    print(df_summary[op_cols].to_string(index=False))


# ── LaTeX summary table (mean ± std, one row per operating point) ─────────────
#
#   Columns : origins (FPR on test) | static_midvholo | photo_replacement
#             | static_midvdynattack | swap_midvdynattack | dynamic_midvdynattack
#   Rows    : ModelName@99 recall
#             ModelName@95 recall
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "HoloVerif-Span"

# Column spec: display header  →  (df column prefix, operating-point tag)
COL_SPEC = [
    ("origins",            "fpr_test"),           # FPR on test set
    ("static midvholo",    "recall_static_midvholo"),
    ("photo-rep",          "recall_photo_replacement"),
    ("static dynattack",   "recall_static_midvdynattack"),
    ("swap dynattack",     "recall_swap_midvdynattack"),
    ("dynamic dynattack",  "recall_dynamic_midvdynattack"),
]


def fmt(series, pct=True):
    """mean ± std across folds, expressed in %."""
    scale = 100 if pct else 1
    return f"{series.mean()*scale:.1f} $\\pm$ {series.std()*scale:.1f}"


# Build table rows
tex_rows = []
for tag in OPERATING_POINTS:
    label = f"{MODEL_NAME}{tag}"
    cells = []
    for _header, col_prefix in COL_SPEC:
        col = f"{col_prefix}{tag}"
        if col in df.columns:
            cells.append(fmt(df[col]))
        else:
            cells.append("--")
    tex_rows.append((label, cells))


# ── Render LaTeX ──────────────────────────────────────────────────────────────
col_headers = [h for h, _ in COL_SPEC]
n_cols = len(col_headers)

lines = []
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\renewcommand{\arraystretch}{1.2}")
lines.append(r"\begin{tabular}{ll" + "r" * n_cols + "}")
lines.append(r"\toprule")
lines.append(
    r"\textbf{Model} & \textbf{Metric} & "
    + " & ".join(f"\\textbf{{{h}}}" for h in col_headers)
    + r" \\"
)
lines.append(r"\midrule")

for label, cells in tex_rows:
    lines.append(
        f"{label} & recall & "
        + " & ".join(cells)
        + r" \\"
    )

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\caption{Recall at operating points @99 and @95 (mean $\pm$ std over 5 folds). ")
lines.append(r"Thresholds are calibrated on the validation set.}")
lines.append(r"\label{tab:recall_operating_points}")
lines.append(r"\end{table}")

latex_table = "\n".join(lines)
print(f"\n{SEP}")
print("LaTeX summary table:")
print(latex_table)
