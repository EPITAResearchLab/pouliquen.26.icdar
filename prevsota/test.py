# A logger for this file
import json
import logging
import os
import random
import shutil
import tempfile
import uuid

import hydra
import mlflow
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from src.utils.utils import (
    already_run,
    get_metrics,
    evaluate_dataset_with_scores,
    mlruntodict,
)

log = logging.getLogger(__name__)


def seed_everything(seed, workers=False):
    """Set seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if workers:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ===========================================================================
# Helpers
# ===========================================================================

def _auc_vs_origins(origins_scores: np.ndarray, fraud_scores: np.ndarray) -> float:
    """Compute AUROC: origins = negative class (0), fraud = positive class (1).

    Single source-of-truth used by every AUC computation in this file.
    Identical to the notebook cell logic:
        y_true  = [0]*len(origins) + [1]*len(fraud)
        y_score = concat(origins, fraud)
        roc_auc_score(y_true, y_score)
    """
    if len(origins_scores) == 0 or len(fraud_scores) == 0:
        log.warning("Empty scores array passed to _auc_vs_origins — returning NaN")
        return float("nan")
    y_true  = np.array([0] * len(origins_scores) + [1] * len(fraud_scores))
    y_score = np.concatenate([origins_scores, fraud_scores])
    return float(roc_auc_score(y_true, y_score))


def _weighted_combined_auc(
    auc_per_dataset: dict[str, float],
    n_per_dataset: dict[str, int],
) -> float:
    """Weighted average AUC across fraud datasets, weighted by dataset length.

    Uses the actual number of FRAUD samples in each dataset as the weight,
    consistent with merge_metrics and the notebook evaluation cell.
    NaN datasets are skipped with a warning.
    """
    total_weight, weighted_sum = 0, 0.0
    for ds, auc_val in auc_per_dataset.items():
        if np.isnan(auc_val):
            log.warning(f"Skipping '{ds}' in combined AUC — AUC is NaN")
            continue
        n = n_per_dataset.get(ds, 0)
        if n == 0:
            log.warning(f"Skipping '{ds}' in combined AUC — n=0")
            continue
        weighted_sum += n * auc_val
        total_weight += n
    return weighted_sum / total_weight if total_weight > 0 else float("nan")


# ===========================================================================
# Metric merging
# ===========================================================================

def merge_metrics(metrics_list, dataset_sizes):
    """Merge metrics from multiple datasets weighted by dataset size."""
    if not metrics_list or not dataset_sizes or len(metrics_list) != len(dataset_sizes):
        if not metrics_list:
            return {}
        return {
            key: sum(m.get(key, 0) for m in metrics_list) / len(metrics_list)
            for key in set().union(*[set(m.keys()) for m in metrics_list])
        }
    all_keys      = set().union(*[set(m.keys()) for m in metrics_list])
    total_samples = sum(dataset_sizes)
    merged = {}
    for key in all_keys:
        wsum, wcount = 0.0, 0.0
        for i, metrics in enumerate(metrics_list):
            if key in metrics:
                w      = dataset_sizes[i] / total_samples
                wsum  += metrics[key] * w
                wcount += w
        if wcount > 0:
            merged[key] = wsum
    return merged


# ===========================================================================
# ROC data helpers
# ===========================================================================

def _build_roc_data(
    origins_scores: np.ndarray,
    fraud_scores_per_dataset: dict[str, np.ndarray],
) -> dict:
    """Build a serialisable dict holding all raw scores needed to (re)plot ROC curves.

    Structure
    ---------
    {
        "origins": [float, ...],                  # legit sample scores (negative class)
        "datasets": {
            "<dataset_name>": [float, ...],        # fraud scores for that dataset
            ...
        }
    }

    Origins are stored separately from every fraud dataset so that any
    downstream notebook can trivially reconstruct the per-dataset ROC curve:

        y_true  = [0]*len(origins) + [1]*len(fraud_ds)
        y_score = origins + fraud_ds
        fpr, tpr, _ = roc_curve(y_true, y_score)
    """
    return {
        "origins": origins_scores.tolist(),
        "datasets": {
            ds: scores.tolist()
            for ds, scores in fraud_scores_per_dataset.items()
        },
    }


def _save_roc_data(
    roc_data: dict,
    temp_dir: str,
    filename: str = "roc_data.json",
) -> str:
    """Serialise *roc_data* as JSON and return the file path."""
    path = os.path.join(temp_dir, filename)
    with open(path, "w") as f:
        json.dump(roc_data, f)
    log.info("ROC data saved to %s", path)
    return path


def _save_roc_data_npz(
    origins_scores: np.ndarray,
    fraud_scores_per_dataset: dict[str, np.ndarray],
    temp_dir: str,
    filename: str = "roc_data.npz",
) -> str:
    """Serialise raw scores as a compressed NumPy archive and return the file path.

    The .npz contains:
        arr_origins              — 1-D float array of legit (origins) scores
        arr_<dataset_name>       — 1-D float array of fraud scores per dataset

    Loading example
    ---------------
        data = np.load("roc_data.npz")
        origins = data["arr_origins"]
        fraud   = data["arr_swap"]          # or whichever dataset key
        fpr, tpr, _ = roc_curve(
            [0]*len(origins) + [1]*len(fraud),
            np.concatenate([origins, fraud]),
        )
    """
    arrays = {"arr_origins": origins_scores}
    for ds, scores in fraud_scores_per_dataset.items():
        # np.savez keys must be valid identifiers — replace hyphens/spaces
        safe_key = "arr_" + ds.replace("-", "_").replace(" ", "_")
        arrays[safe_key] = scores

    path = os.path.join(temp_dir, filename)
    np.savez_compressed(path, **arrays)
    log.info("ROC data (npz) saved to %s", path)
    return path


def _save_roc_curves(
    origins_scores: np.ndarray,
    fraud_scores_per_dataset: dict[str, np.ndarray],
    auc_per_dataset: dict[str, float],
    temp_dir: str,
    pipeline: str = "deep_learning",
    score_name: str = "reconstruction anomaly score",
) -> tuple[str, str]:
    """Compute and save precomputed ROC curves (fpr, tpr, thresholds) per dataset.

    Produces two files alongside roc_data.{npz,json}:

    roc_curves.npz  — compressed NumPy archive.
                      Keys per dataset (hyphens → underscores):
                        fpr_<ds>   — false positive rates
                        tpr_<ds>   — true positive rates
                        thr_<ds>   — decision thresholds

    roc_curves.json — human-readable manifest.
                      Contains pipeline metadata + per-dataset dict with
                      auc, n_fraud, n_origins, fpr, tpr, thresholds.

    Convention (identical to gmm pipeline):
        y_true  = [0]*n_origins + [1]*n_fraud
        y_score = concat(origins, fraud)
        → higher score == more anomalous (fraud)
    """
    npz_arrays   : dict[str, np.ndarray] = {}
    json_datasets: dict[str, dict]       = {}

    for ds, fraud_scores in fraud_scores_per_dataset.items():
        y_true  = np.array([0] * len(origins_scores) + [1] * len(fraud_scores))
        y_score = np.concatenate([origins_scores, fraud_scores])
        fpr, tpr, thr = roc_curve(y_true, y_score)

        safe = ds.replace("-", "_").replace(" ", "_")
        npz_arrays[f"fpr_{safe}"] = fpr
        npz_arrays[f"tpr_{safe}"] = tpr
        npz_arrays[f"thr_{safe}"] = thr

        json_datasets[ds] = {
            "auc":        auc_per_dataset.get(ds, float("nan")),
            "n_fraud":    int(len(fraud_scores)),
            "n_origins":  int(len(origins_scores)),
            "fpr":        fpr.tolist(),
            "tpr":        tpr.tolist(),
            "thresholds": thr.tolist(),
        }

    npz_path = os.path.join(temp_dir, "roc_curves.npz")
    np.savez_compressed(npz_path, **npz_arrays)
    log.info("ROC curves (npz) saved to %s", npz_path)

    manifest = {
        "pipeline":        pipeline,
        "score_name":      score_name,
        "score_direction": "higher_is_more_anomalous",
        "datasets":        json_datasets,
    }
    json_path = os.path.join(temp_dir, "roc_curves.json")
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("ROC curves (json) saved to %s", json_path)

    return npz_path, json_path

def _save_roc_curves_grouped(
    origins_scores: np.ndarray,
    fraud_scores_per_dataset: dict[str, np.ndarray],
    dataset_groups: dict[str, list[str]],
    temp_dir: str,
    pipeline: str = "deep_learning",
    score_name: str = "reconstruction anomaly score",
) -> tuple[str, str]:
    """Compute and save group-level ROC curves by concatenating fraud scores.

    For each group:
        - pools all available fraud datasets in that group
        - calls roc_curve(origins + pooled_fraud) once  ← exact, no interpolation
        - stores fpr / tpr / thresholds in npz + json

    Files produced:
        roc_curves_grouped.npz   — keys: fpr_<group>, tpr_<group>, thr_<group>
        roc_curves_grouped.json  — human-readable manifest with auc per group
    """
    npz_arrays   : dict[str, np.ndarray] = {}
    json_groups  : dict[str, dict]       = {}

    for group_name, datasets in dataset_groups.items():
        available = [ds for ds in datasets if ds in fraud_scores_per_dataset]
        if not available:
            log.warning("Group ROC: skipping '%s' — no datasets available", group_name)
            continue

        # Concatenate all fraud scores in this group (exact aggregate)
        pooled_fraud = np.concatenate([fraud_scores_per_dataset[ds] for ds in available])

        y_true  = np.array([0] * len(origins_scores) + [1] * len(pooled_fraud))
        y_score = np.concatenate([origins_scores, pooled_fraud])

        fpr, tpr, thr = roc_curve(y_true, y_score)
        auc_val = float(roc_auc_score(y_true, y_score))

        safe = group_name.replace("-", "_").replace(" ", "_")
        npz_arrays[f"fpr_{safe}"] = fpr
        npz_arrays[f"tpr_{safe}"] = tpr
        npz_arrays[f"thr_{safe}"] = thr

        json_groups[group_name] = {
            "auc":          auc_val,
            "n_fraud":      int(len(pooled_fraud)),
            "n_origins":    int(len(origins_scores)),
            "datasets_used": available,
            "fpr":          fpr.tolist(),
            "tpr":          tpr.tolist(),
            "thresholds":   thr.tolist(),
        }

        log.info(
            "  ROC group '%s': AUC=%.4f  (n_fraud=%d, datasets=%s)",
            group_name, auc_val, len(pooled_fraud), available,
        )

    npz_path = os.path.join(temp_dir, "roc_curves_grouped.npz")
    np.savez_compressed(npz_path, **npz_arrays)

    manifest = {
        "pipeline":        pipeline,
        "score_name":      score_name,
        "score_direction": "higher_is_more_anomalous",
        "groups":          json_groups,
    }
    json_path = os.path.join(temp_dir, "roc_curves_grouped.json")
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return npz_path, json_path

# ===========================================================================
# Main pipeline
# ===========================================================================

def pipeline_with_roc(cfg, params, task_name_full, log, dataset_groups=None):
    """Evaluate the model and log AUC metrics to MLflow.

    IMPORTANT — mixed first dataset
    --------------------------------
    The first dataset in cfg.data.test may contain BOTH legit (isFraud=False)
    and fraud (isFraud=True) samples (e.g. static_midvholo shares its split
    with the origins).  evaluate_dataset_with_scores already labels every
    sample correctly via dataset.isFraud(i), so we:

        origins_scores = y_score[y_true == 0]   ← legit samples only
        fraud_scores   = y_score[y_true == 1]   ← fraud samples in same file

    This means:
    * origins_scores is ALWAYS pure legit, regardless of which dataset it
      comes from.
    * If the first dataset contains fraud samples, they get their own AUC
      entry under that dataset name.
    * All subsequent datasets are expected to be pure fraud (y_true all 1),
      but the same filter is applied defensively.

    AUC computation
    ---------------
    All AUC values use _auc_vs_origins(origins_scores, fraud_scores), which
    calls roc_auc_score(y_true=[0..1..], y_score) — no threshold search,
    no FPR/TPR interpolation, no common-FPR grid.
    Combined AUC = _weighted_combined_auc weighted by number of fraud samples.

    ROC data artifacts
    ------------------
    Two artifacts are logged under the ``roc_data/`` MLflow artifact folder:

    * ``roc_data.json``  — human-readable; origins list + per-dataset fraud lists.
    * ``roc_data.npz``   — compressed NumPy archive; same data, faster to load
                           for plotting.  Key convention: ``arr_origins`` for
                           legit scores, ``arr_<dataset>`` for fraud scores
                           (hyphens → underscores).
    """
    run_uuid = str(uuid.uuid4())
    temp_dir = os.path.join(tempfile.gettempdir(), f"mlflow_run_{run_uuid}")
    os.makedirs(temp_dir, exist_ok=True)
    log.info(f"Created temporary directory: {temp_dir}")

    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            log.info(f"Started MLflow run: {run_id}")

            mlflow.set_tag("mlflow.runName", task_name_full)
            mlflow.log_params(params)

            all_metrics:          dict[str, dict]       = {}
            all_raw_scores:       dict[str, dict]       = {}
            dataset_sample_sizes: dict[str, int]        = {}
            auc_per_dataset:      dict[str, float]      = {}  # fraud ds → scalar AUROC
            n_fraud_per_dataset:  dict[str, int]        = {}  # fraud ds → n fraud samples

            # Fraud scores keyed by dataset name — populated during inference
            # loop, used to build ROC data artifacts after AUC computation.
            fraud_scores_per_dataset: dict[str, np.ndarray] = {}

            test_datasets  = list(cfg.data.test.keys())

            # origins_scores is populated from ALL legit samples seen across
            # every dataset (defensive: in practice only dataset[0] has legits)
            origins_scores: np.ndarray = np.array([])

            # ------------------------------------------------------------------
            # 1. Inference loop
            # ------------------------------------------------------------------
            for i, test_d in enumerate(test_datasets):
                seed_everything(cfg.seed, workers=True)
                model     = instantiate(params.model)
                decision  = instantiate(params.decision)
                data_test = instantiate(cfg.data.test[test_d])

                log.info("Processing dataset '%s' at %s", test_d, data_test.input_dir)

                metrics, y_true, y_score = evaluate_dataset_with_scores(
                    data_test, model, decision
                )
                all_metrics[test_d]          = metrics
                dataset_sample_sizes[test_d] = len(data_test)
                all_raw_scores[test_d]       = {"y_true": y_true, "y_score": y_score}

                # Log standard per-dataset classification metrics
                mlflow.log_metrics(
                    {f"{test_d}_{k}": float(v) for k, v in metrics.items()}
                )

                # --- Split legit vs fraud using the ground-truth label --------
                mask_legit = y_true == 0
                mask_fraud = y_true == 1

                legit_scores_here = y_score[mask_legit]
                fraud_scores_here = y_score[mask_fraud]

                n_legit = int(mask_legit.sum())
                n_fraud = int(mask_fraud.sum())

                log.info(
                    "  '%s': %d legit, %d fraud samples",
                    test_d, n_legit, n_fraud,
                )

                # Accumulate origins (legit) scores
                if n_legit > 0:
                    origins_scores = (
                        np.concatenate([origins_scores, legit_scores_here])
                        if len(origins_scores) > 0
                        else legit_scores_here
                    )

                # Store per-dataset fraud scores for ROC artifact
                if n_fraud > 0:
                    fraud_scores_per_dataset[test_d] = fraud_scores_here
                    auc_per_dataset[test_d]           = None  # placeholder; filled below
                    n_fraud_per_dataset[test_d]       = n_fraud

            log.info(
                "Origins pool: %d legit samples from %d dataset(s)",
                len(origins_scores),
                sum(1 for d in test_datasets
                    if all_raw_scores[d]["y_true"].sum() < len(all_raw_scores[d]["y_true"])),
            )

            # ------------------------------------------------------------------
            # 2. Compute per-dataset AUC now that origins_scores is complete
            # ------------------------------------------------------------------
            for test_d in list(auc_per_dataset.keys()):
                fraud_scores_here = fraud_scores_per_dataset[test_d]
                auc_val = _auc_vs_origins(origins_scores, fraud_scores_here)
                auc_per_dataset[test_d] = auc_val
                mlflow.log_metric(f"{test_d}_auc", auc_val)
                log.info(
                    "  AUC '%s': %.4f  (n_fraud=%d, n_origins=%d)",
                    test_d, auc_val,
                    n_fraud_per_dataset[test_d], len(origins_scores),
                )

            # ------------------------------------------------------------------
            # 3. Combined AUC — weighted by number of fraud samples
            # ------------------------------------------------------------------
            auc_combined = _weighted_combined_auc(auc_per_dataset, n_fraud_per_dataset)
            mlflow.log_metric("auc_combined", auc_combined)
            log.info("Combined AUC (weighted by n_fraud): %.4f", auc_combined)

            # ------------------------------------------------------------------
            # 4. Save AUC summary artifact
            # ------------------------------------------------------------------
            auc_summary = {
                "auc_per_dataset":     auc_per_dataset,
                "auc_combined":        auc_combined,
                "n_fraud_per_dataset": n_fraud_per_dataset,
                "n_origins":           int(len(origins_scores)),
                "run_id":              run_id,
                "seed":                cfg.seed,
            }
            auc_path = os.path.join(temp_dir, "auc_scores.json")
            with open(auc_path, "w") as f:
                json.dump(auc_summary, f, indent=2)
            mlflow.log_artifact(auc_path, "auc_data")

            # ------------------------------------------------------------------
            # 5. Save ROC raw-score artifacts
            #    • roc_data.json  — human-readable, origins separated from frauds
            #    • roc_data.npz   — compressed NumPy archive, same layout
            #
            #    Both files are logged under the "roc_data/" artifact folder so
            #    they can be retrieved in one call:
            #        mlflow.artifacts.download_artifacts(run_id=..., artifact_path="roc_data")
            # ------------------------------------------------------------------
            roc_data = _build_roc_data(origins_scores, fraud_scores_per_dataset)

            roc_json_path = _save_roc_data(roc_data, temp_dir)
            mlflow.log_artifact(roc_json_path, "roc_data")

            roc_npz_path = _save_roc_data_npz(
                origins_scores, fraud_scores_per_dataset, temp_dir
            )
            mlflow.log_artifact(roc_npz_path, "roc_data")

            log.info(
                "ROC artifacts logged: %d origins, %d fraud datasets (%s)",
                len(origins_scores),
                len(fraud_scores_per_dataset),
                ", ".join(fraud_scores_per_dataset.keys()),
            )

            # ------------------------------------------------------------------
            # 5b. Save precomputed ROC curves (fpr / tpr / thresholds per dataset)
            #     Logged alongside roc_data.* under the same "roc_data/" folder.
            # ------------------------------------------------------------------
            roc_curves_npz, roc_curves_json = _save_roc_curves(
                origins_scores,
                fraud_scores_per_dataset,
                auc_per_dataset,
                temp_dir,
            )
            mlflow.log_artifact(roc_curves_npz,  "roc_data")
            mlflow.log_artifact(roc_curves_json, "roc_data")
            log.info(
                "ROC curves logged: %d fraud datasets (%s)",
                len(fraud_scores_per_dataset),
                ", ".join(fraud_scores_per_dataset.keys()),
            )

            # ------------------------------------------------------------------
            # 6. Dataset groups
            # ------------------------------------------------------------------
            if dataset_groups:
                group_auc_per_group = {}
                group_n_per_group   = {}

                for group_name, datasets in dataset_groups.items():
                    available = [ds for ds in datasets if ds in all_raw_scores]
                    if not available:
                        log.warning("Skipping group '%s': no datasets loaded", group_name)
                        continue

                    # Merge standard metrics (weighted by total dataset size)
                    group_merged = merge_metrics(
                        [all_metrics[ds] for ds in available],
                        [dataset_sample_sizes[ds] for ds in available],
                    )
                    for k, v in group_merged.items():
                        mlflow.log_metric(f"{group_name}_{k}", float(v))

                    # Pool FRAUD-ONLY scores across all datasets in the group
                    merged_fraud_scores = np.concatenate([
                        fraud_scores_per_dataset[ds]
                        for ds in available
                        if ds in fraud_scores_per_dataset
                    ])
                    group_n = int(len(merged_fraud_scores))

                    if group_n == 0:
                        log.warning("Skipping group '%s': 0 fraud samples", group_name)
                        continue

                    group_auc = _auc_vs_origins(origins_scores, merged_fraud_scores)
                    mlflow.log_metric(f"{group_name}_auc", group_auc)
                    log.info(
                        "  AUC group '%s': %.4f  (n_fraud=%d)",
                        group_name, group_auc, group_n,
                    )

                    group_auc_per_group[group_name] = group_auc
                    group_n_per_group[group_name]   = group_n

                # Combined AUC across groups
                if group_auc_per_group:
                    group_combined = _weighted_combined_auc(
                        group_auc_per_group, group_n_per_group
                    )
                    mlflow.log_metric("auc_combined_groups", group_combined)
                    log.info("Combined AUC across groups: %.4f", group_combined)

                roc_grp_npz, roc_grp_json = _save_roc_curves_grouped(
                    origins_scores,
                    fraud_scores_per_dataset,
                    dataset_groups,
                    temp_dir,
                )
                mlflow.log_artifact(roc_grp_npz, "roc_data")
                mlflow.log_artifact(roc_grp_json, "roc_data")
                log.info("Group ROC curves logged for groups: %s", list(dataset_groups.keys()))

    finally:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                log.info("Cleaned up temporary directory: %s", temp_dir)
        except Exception as e:
            log.warning("Error cleaning up temporary directory: %s", e)


# ===========================================================================
# Dataset groups
# ===========================================================================

dataset_groups = {
    "static template swaping": [
        "swap",
        "swap_three",
    ],
    "dynamic template": [
        "double_sticker",
        "holo-completemask",
        "holo-star-horld",
        "leaf-holo",
        "plain-holo",
    ],
    "static template": [
        "laser",
        "no-holo",
        "plastified-led",
        "plastified-lowreflect",
        "plastified-noholo",
    ],
    "MIDV-DynAttack": [
        "swap",
        "swap_three",
        "double_sticker",
        "holo-completemask",
        "holo-star-horld",
        "leaf-holo",
        "plain-holo",
        "laser",
        "no-holo",
        "plastified-led",
        "plastified-lowreflect",
        "plastified-noholo",
    ],
    "MIDV_Holo": ["midv-holo-test"],
    "photo-replacement": ["midv-holo-photo-replacement"]
    # "MIDV_DynAttack": [
    #     "swap",
    #     "swap_three",
    #     "double_sticker",
    #     "holo-completemask",
    #     "holo-star-horld",
    #     "leaf-holo",
    #     "plain-holo",
    #     "laser",
    #     "no-holo",
    #     "plastified-led",
    #     "plastified-lowreflect",
    #     "plastified-noholo",
    # ],
}


@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    """Entry point for testing."""
    seed_everything(cfg.seed, workers=True)
    task_name_full = (
        HydraConfig.get().runtime.choices.decision + "_" + cfg.task_name
    )
    if cfg.get("training", ""):
        task_name_full += cfg.training.trainer.run_name
    cfg.tuner.experiment_name = task_name_full
    tuner = instantiate(cfg.tuner)

    run   = tuner.get_best_run()
    params = mlruntodict(run.data.params)
    params = DictConfig(params)

    mlflow.set_experiment("test5_" + cfg.paths.split_name)
    pipeline_with_roc(cfg, params, task_name_full, log, dataset_groups)

if __name__ == "__main__":
    main()
