"""Evaluation and OOD detection script for Hologram Fraud Detection.

Handles all model types (Transformer, GRU, Hybrid, VAE).
VAE returns (output, mu, logvar) — the others return a single tensor.

OOD scoring signals:
  - All models:  reconstruction MSE, cosine distance, max-frame error
  - VAE only:    KL divergence, ELBO (combined reconstruction + KL)

Metrics:
  - Per-sequence loss statistics
  - Per-video aggregated loss statistics (mean of sequences per video)
  - Threshold-based OOD rate  (percentiles from validation / in-distribution data)
  - AUROC  (test origins vs. each fraud type, per signal)

Thresholds (p95, p99) are derived from the **validation** set.
AUROCs are derived from the **test** set only (origins-test vs fraud-test).

Plots (saved as SVG):
  Per-sequence:
    - Violin plots for MSE, cosine, max-frame-error (and KL/ELBO for VAE)
    - Scatter plot of cosine distance vs reconstruction MSE
    - Density hex-bin plot of cosine vs MSE
    - AUROC bar chart per signal per fraud type
    - Pairwise origins-vs-fraud scatter (cosine vs MSE, one per fraud)
        → saved in per_sequence/origins_vs_fraud/

  Per-video (aggregated):
    - Violin plots for video-aggregated MSE, cosine, max-frame-error (and KL/ELBO)
    - Scatter plot of video-aggregated cosine vs MSE
    - AUROC bar chart per signal per fraud type (video-level)
    - Strip plot showing individual video scores within each fraud type
    - Pairwise origins-vs-fraud scatter (cosine vs MSE, one per fraud)
        → saved in per_video/origins_vs_fraud/
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from configs.config import Config
from eval.ocsvm import compute_stats_all_levels as compute_stats_all_levels_ocsvm
from eval.gmm import fit_gmm_and_score, compute_stats_all_levels, print_stats_summary, auroc
from eval.ocsvm import fit_ocsvm_and_score
from eval.scores import aggregate_scores_by_video, compute_anomaly_scores, is_vae
from models.models_init import create_model
from viz.plot import plot_auroc_bar, plot_cosine_reco_vs_cosine_seq, plot_mse_reco_vs_seq, plot_origins_vs_fraud, plot_video_strip, plot_violin
from viz.plot_gmm import plot_gmm_for_all_frauds, plot_gmm_grouped, plot_auroc_bar_grouped
from eval.thresholds import compute_reference_thresholds, evaluate_ood
from eval.grouping import aggregate_to_groups, merge_id_passport_variants
from viz.plot_ocsvm import plot_ocsvm_for_all_frauds, plot_ocsvm_grouped, plot_ocsvm_val_vs_test

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ──────────────────────────────────────────────────────────────
# Video-level aggregation
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_data(data_path: str):
    return pickle.load(Path(data_path).open("rb"))


def _build_video_names(seq_dict, split_keys):
    """Build a list of video names parallel to the concatenated sequences.

    For each video key k in split_keys that exists in seq_dict,
    repeat str(k) for the number of sequences in seq_dict[k].
    """
    video_names = []
    for k in split_keys:
        if k in seq_dict:
            n = seq_dict[k].shape[0]
            video_names.extend([str(k)] * n)
    return video_names


def get_dataloaders(sequences, sequences_fakeholo, splits, batch_size=32):
    split_train, split_val, split_test = splits

    # ── Train ─────────────────────────────────────────────────
    train_data = torch.cat(
        [sequences[k] for k in split_train if k in sequences], dim=0
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False
    )

    # ── Validation ────────────────────────────────────────────
    val_data = torch.cat(
        [sequences[k] for k in split_val if k in sequences], dim=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )
    val_video_names = _build_video_names(sequences, split_val)

    # ── Test (origins) ────────────────────────────────────────
    test_data = torch.cat(
        [sequences[k] for k in split_test if k in sequences], dim=0
    )
    test_loader_holo = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )
    test_video_names = _build_video_names(sequences, split_test)

    # ── Test (fakes) ──────────────────────────────────────────
    test_loaders_fake = []
    test_video_names_fake = []
    for seq_fake in sequences_fakeholo:
        fd = torch.cat(
            [seq_fake[k] for k in split_test if k in seq_fake], dim=0
        )
        test_loaders_fake.append(
            torch.utils.data.DataLoader(
                fd, batch_size=batch_size, shuffle=False
            )
        )
        test_video_names_fake.append(
            _build_video_names(seq_fake, split_test)
        )

    return (
        train_loader,
        val_loader,
        test_loader_holo,
        test_loaders_fake,
        val_video_names,
        test_video_names,
        test_video_names_fake,
    )


# ──────────────────────────────────────────────────────────────
# Plot orchestration
# ──────────────────────────────────────────────────────────────

def generate_all_plots(
    all_scores: dict[str, dict[str, np.ndarray]],
    all_results: dict[str, dict],
    all_video_scores: dict[str, dict[str, np.ndarray]],
    all_video_ids: dict[str, list[str]],
    all_video_results: dict[str, dict],
    val_thresholds: dict[str, dict],
    val_thresholds_video: dict[str, dict],
    output_dir: str,
    vae: bool,
):
    """Generate and save all diagnostic plots (per-sequence + per-video)."""
    out = Path(output_dir)
    seq_dir = out / "per_sequence"
    vid_dir = out / "per_video"
    seq_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    base_signals = [
        ("recon_mse", "Reconstruction MSE", "MSE"),
        ("recon_cosine", "Cosine Distance", "1 − cos(output, target)"),
        ("max_frame_error", "Max Frame Error", "Max single-frame MSE"),
    ]
    vae_signals = [
        ("kl_divergence", "KL Divergence", "KL( q(z|x) ‖ p(z) )"),
        ("elbo", "ELBO (MSE + 0.1·KL)", "ELBO"),
    ]
    signals = base_signals + (vae_signals if vae else [])

    # ══════════════════════════════════════════════════════════
    # PER-SEQUENCE PLOTS
    # ══════════════════════════════════════════════════════════
    print("\n── Generating per-sequence plots ───────────────────")

    for key, title, ylabel in signals:
        plot_violin(
            all_scores,
            signal_key=key,
            title=f"{title} per Fraud Type (per sequence)",
            ylabel=ylabel,
            out_path=str(seq_dir / f"violin_{key}.svg"),
            ref_thresholds=val_thresholds,
        )
    
    plot_cosine_reco_vs_cosine_seq(
        all_scores,
        out_path=str(seq_dir / "scatter_cosinereco_vs_cosineseq.svg"),
        ref_thresholds=val_thresholds,
    )

    plot_mse_reco_vs_seq(
        all_scores,
        out_path=str(seq_dir / "density_cosine_vs_mse.svg"),
        ref_thresholds=val_thresholds,
    )

    plot_auroc_bar(
        all_results,
        out_path=str(seq_dir / "bar_auroc_per_signal.svg"),
        title="AUROC per Signal per Fraud Type (per sequence)",
    )

    # ── Origins vs each fraud (sequence-level) ────────────────
    seq_fraud_dir = seq_dir / "origins_vs_fraud"
    plot_origins_vs_fraud(
        all_scores,
        out_dir=str(seq_fraud_dir),
        ref_thresholds=val_thresholds,
        level="sequence",
    )

    # ══════════════════════════════════════════════════════════
    # PER-VIDEO PLOTS
    # ══════════════════════════════════════════════════════════
    print("\n── Generating per-video plots ──────────────────────")

    for key, title, ylabel in signals:
        plot_violin(
            all_video_scores,
            signal_key=key,
            title=f"{title} per Fraud Type (per video)",
            ylabel=ylabel,
            out_path=str(vid_dir / f"violin_{key}.svg"),
            ref_thresholds=val_thresholds_video,
        )

        plot_video_strip(
            all_video_scores,
            all_video_ids,
            signal_key=key,
            title=f"{title} — Individual Videos",
            ylabel=ylabel,
            out_path=str(vid_dir / f"strip_{key}.svg"),
            ref_thresholds=val_thresholds_video,
        )
    # scatter_cosinereco_vs_cosineseq
    plot_mse_reco_vs_seq(
        all_video_scores,
        out_path=str(vid_dir / "scatter_cosine_vs_mse.svg"),
        ref_thresholds=val_thresholds_video,
        # max_points_per_class=5000,
    )

    plot_auroc_bar(
        all_video_results,
        out_path=str(vid_dir / "bar_auroc_per_signal.svg"),
        title="AUROC per Signal per Fraud Type (per video)",
    )

    # ── Origins vs each fraud (video-level) ───────────────────
    vid_fraud_dir = vid_dir / "origins_vs_fraud"
    plot_origins_vs_fraud(
        all_video_scores,
        out_dir=str(vid_fraud_dir),
        ref_thresholds=val_thresholds_video,
        max_points_per_class=5000,
        level="video",
    )


def run_grouped_analysis(
    all_scores: dict,
    gmm_results: dict,
    val_thresholds: dict,
    output_dir: str,
    level_name: str = "sequence",
    val_scores:dict = {},
):
    """Run full grouped analysis pipeline."""
    
    # 1. Merge _ID/_passport variants
    merged_scores = merge_id_passport_variants(all_scores)
    print(f"\n  Merged {len(all_scores)} variants → {len(merged_scores)} base types")
    
    # 2. Aggregate to groups
    grouped_scores = aggregate_to_groups(merged_scores)
    print(f"  Grouped into {len(grouped_scores) - 1} attack categories + origins")
    
    # 3. Compute stats at all levels
    stats = compute_stats_all_levels(
        all_scores=all_scores,
        gmm_results=gmm_results,
        val_thresholds=val_thresholds,
        auroc_fn=auroc,  # your auroc function
    )
    
    # 4. Print summary
    print_stats_summary(stats, level_name=level_name)
    
    # 5. Generate plots
    gmm_dir = Path(output_dir) / f"per_{level_name}" / "gmm"
    
    plot_gmm_grouped(
        gmm_results=gmm_results,
        grouped_scores=grouped_scores,
        origins_scores=all_scores["origins"],
        stats=stats,
        level_name=level_name,
        out_dir=gmm_dir,
        val_scores=val_scores,

    )
    
    plot_auroc_bar_grouped(
        stats=stats,
        level_name=level_name,
        out_dir=gmm_dir,
    )
    
    # 6. Save stats JSON
    import json
    stats_path = gmm_dir / f"grouped_stats_{level_name}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"  ✓ Saved {stats_path}")
    
    return stats


def run_grouped_analysis_ocsvm(
    all_scores: dict,
    ocsvm_results: dict,
    val_thresholds: dict,
    output_dir: str,
    level_name: str = "sequence",
):
    """Run full grouped analysis pipeline."""

    # 1. Merge _ID/_passport variants
    merged_scores = merge_id_passport_variants(all_scores)
    print(f"\n  Merged {len(all_scores)} variants → {len(merged_scores)} base types")

    # 2. Aggregate to groups
    grouped_scores = aggregate_to_groups(merged_scores)
    print(f"  Grouped into {len(grouped_scores) - 1} attack categories + origins")

    # 3. Compute stats at all levels
    stats = compute_stats_all_levels_ocsvm(
        all_scores=all_scores,
        ocsvm_results=ocsvm_results,
        val_thresholds=val_thresholds,
        auroc_fn=auroc,
    )

    # 4. Print summary
    print_stats_summary(stats, level_name=level_name)

    # 5. Generate plots
    ocsvm_dir = Path(output_dir) / f"per_{level_name}" / "ocsvm"

    plot_ocsvm_grouped(
        ocsvm_results=ocsvm_results,
        grouped_scores=grouped_scores,
        origins_scores=all_scores["origins"],
        stats=stats,
        level_name=level_name,
        out_dir=ocsvm_dir,
    )

    plot_auroc_bar_grouped(
        stats=stats,
        level_name=level_name,
        out_dir=ocsvm_dir,
    )

    # 6. Save stats JSON
    stats_path = ocsvm_dir / f"grouped_stats_{level_name}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"  ✓ Saved {stats_path}")

    return stats, grouped_scores

# ──────────────────────────────────────────────────────────────
# Main test loop
# ──────────────────────────────────────────────────────────────

def test(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    val_video_names: list[str],
    test_dataloaders: list[torch.utils.data.DataLoader],
    test_video_names_list: list[list[str]],
    fake_holo_names: list[str],
    device: torch.device,
    output_dir: str = "./plots",
) -> dict:
    """Full evaluation pipeline.

    1. Score the validation set → thresholds only (p95, p99).
    2. Score every test set (origins + each fraud type).
    3. Compute OOD metrics:  AUROC uses test-origins vs test-fraud,
       OOD rates use validation thresholds.
    4. Aggregate to video level and repeat.
    5. Generate diagnostic plots.
    """
    model.to(device).eval()
    vae = is_vae(model)

    # ── 1. validation → thresholds only ───────────────────────
    print("\n[Val] Scoring validation set (for thresholds only) …")
    val_scores = compute_anomaly_scores(model, val_dataloader, device)
    val_thresholds = compute_reference_thresholds(val_scores)

    val_scores_video, val_video_ids = aggregate_scores_by_video(val_scores, val_video_names)
    val_thresholds_video = compute_reference_thresholds(val_scores_video)

    print(f"  Model type : {'VAE' if vae else 'deterministic'}")
    print(f"  Signals    : {list(val_scores.keys())}")
    print(f"  Val sequences: {len(next(iter(val_scores.values())))},  "
          f"Val videos: {len(val_video_ids)}")
    for sig, th in val_thresholds.items():
        th_v = val_thresholds_video[sig]
        print(f"  {sig:20s}  seq: p95={th['percentiles']['p95']:.6f} p99={th['percentiles']['p99']:.6f}  "
              f"| vid: p95={th_v['percentiles']['p95']:.6f} p99={th_v['percentiles']['p99']:.6f}")

    # ── 2. score every test set ───────────────────────────────
    print("\n[Test] Scoring all test sets …")

    # We need origins scored first so we can use it as the ID
    # reference for AUROC computation.
    # fake_holo_names[0] is expected to be "origins".
    assert fake_holo_names[0] == "origins", (
        f"First entry in fake_holo_names must be 'origins', got '{fake_holo_names[0]}'"
    )

    all_scores: dict[str, dict[str, np.ndarray]] = {}
    all_video_scores: dict[str, dict[str, np.ndarray]] = {}
    all_video_ids: dict[str, list[str]] = {}

    for loader, video_names, name in zip(
        test_dataloaders, test_video_names_list, fake_holo_names
    ):
        scores = compute_anomaly_scores(model, loader, device)
        all_scores[name] = scores

        vid_scores, vid_ids = aggregate_scores_by_video(scores, video_names)
        all_video_scores[name] = vid_scores
        all_video_ids[name] = vid_ids

    # ── 3. OOD evaluation (AUROC from test origins) ──────────
    origins_seq_scores = all_scores["origins"]
    origins_vid_scores = all_video_scores["origins"]

    all_results = {}
    all_video_results = {}
    header_printed = False

    for name in fake_holo_names:
        # per-sequence
        result = evaluate_ood(
            id_scores=origins_seq_scores,
            val_thresholds=val_thresholds,
            test_scores=all_scores[name],
            name=name,
        )
        all_results[name] = result

        # per-video
        vid_result = evaluate_ood(
            id_scores=origins_vid_scores,
            val_thresholds=val_thresholds_video,
            test_scores=all_video_scores[name],
            name=name,
        )
        all_video_results[name] = vid_result

        if not header_printed:
            sig_names = list(result["signals"].keys())
            header = f"{'name':30s}  {'#seq':>5s}  {'#vid':>5s}"
            for s in sig_names:
                header += f" | {s + '_auroc':>16s}  {s + '_vid_auroc':>16s}"
            print(f"\n{header}")
            print("─" * len(header))
            header_printed = True

        row = f"{name:30s}  {result['n_samples']:5d}  {len(all_video_ids[name]):5d}"
        for s in sig_names:
            seq_info = result["signals"][s]
            vid_info = vid_result["signals"][s]
            row += f" | {seq_info['auroc']:16.4f}  {vid_info['auroc']:16.4f}"
        print(row)

    # ── 4. summary ────────────────────────────────────────────
    print("\n── Per-fraud-type loss summary (sequence level) ────")
    print("    AUROC = test-origins vs test-fraud  |  OOD rates = val thresholds")
    for name, res in all_results.items():
        r = res["signals"]["recon_mse"]
        extra = ""
        if "kl_divergence" in res["signals"]:
            k = res["signals"]["kl_divergence"]
            extra = f"  kl={k['mean']:.6f}±{k['std']:.6f}"
        print(f"  {name:30s}  mse={r['mean']:.6f}±{r['std']:.6f}  "
              f"auroc={r['auroc']:.4f}  ood@p95={r['ood_rate_p95']:.2%}{extra}")

    print("\n── Per-fraud-type loss summary (video level) ───────")
    print("    AUROC = test-origins vs test-fraud  |  OOD rates = val thresholds")
    for name, res in all_video_results.items():
        r = res["signals"]["recon_mse"]
        n_vids = len(all_video_ids[name])
        extra = ""
        if "kl_divergence" in res["signals"]:
            k = res["signals"]["kl_divergence"]
            extra = f"  kl={k['mean']:.6f}±{k['std']:.6f}"
        print(f"  {name:30s}  [{n_vids:3d} videos]  mse={r['mean']:.6f}±{r['std']:.6f}  "
              f"auroc={r['auroc']:.4f}  ood@p95={r['ood_rate_p95']:.2%}{extra}")

    # ── 5. plots ──────────────────────────────────────────────
    generate_all_plots(
        all_scores=all_scores,
        all_results=all_results,
        all_video_scores=all_video_scores,
        all_video_ids=all_video_ids,
        all_video_results=all_video_results,
        val_thresholds=val_thresholds,
        val_thresholds_video=val_thresholds_video,
        output_dir=output_dir,
        vae=vae,
    )

    # ------------------ GMM ------------------

    # sequence-level GMM
    seq_gmm_dir = Path(output_dir) / 'per_sequence' / 'gmm'
    seq_out = fit_gmm_and_score(
        id_scores=val_scores,                # or origins_seq_scores if you prefer test-origins
        test_scores_map=all_scores,
        val_scores=val_scores,
        level_name='sequence',
        out_dir=seq_gmm_dir,
    )

    # Plot sequence-level GMM for all frauds
    plot_gmm_for_all_frauds(
        gmm_results=seq_out,
        id_scores=val_scores,
        test_scores_map=all_scores,
        level_name='sequence',
        out_dir=seq_gmm_dir,
    )


    # video-level GMM
    vid_gmm_dir = Path(output_dir) / 'per_video' / 'gmm'
    vid_out = fit_gmm_and_score(
        id_scores=val_scores_video,
        test_scores_map=all_video_scores,
        val_scores=val_scores_video,
        level_name='video',
        out_dir=vid_gmm_dir,
    )
    # Plot video-level GMM for all frauds
    plot_gmm_for_all_frauds(
        gmm_results=vid_out,
        id_scores=val_scores_video,
        test_scores_map=all_video_scores,
        level_name='video',
        out_dir=vid_gmm_dir,
    )

    # Print top-level summary
    print('\\nGMM sequence-level: components=', seq_out['gmm'].n_components)
    print('thresholds (anom p95/p99)=', seq_out['thresholds']['p95'], seq_out['thresholds']['p99'])
    print('Example AUROCs (sequence):')
    for k, v in list(seq_out['auroc'].items()):
        print(f"  {k:30s}  AUROC={v:.4f}  OOD@p95={seq_out['ood_rates'][k]['p95']:.2%}  OOD@p99={seq_out['ood_rates'][k]['p99']:.2%}")

    print('\\nGMM video-level: components=', vid_out['gmm'].n_components)
    print('thresholds (anom p95/p99)=', vid_out['thresholds']['p95'], vid_out['thresholds']['p99'])
    print('Example AUROCs (video):')
    for k, v in list(vid_out['auroc'].items()):
        print(f"  {k:30s}  AUROC={v:.4f}  OOD@p95={vid_out['ood_rates'][k]['p95']:.2%}  OOD@p99={vid_out['ood_rates'][k]['p99']:.2%}")


    seq_stats = run_grouped_analysis(
        all_scores=all_scores,
        gmm_results=seq_out,
        val_thresholds=val_thresholds,
        output_dir=output_dir,
        level_name="sequence",
        val_scores=val_scores_video,
    )

    vid_stats = run_grouped_analysis(
        all_scores=all_video_scores,
        gmm_results=vid_out,
        val_thresholds=val_thresholds_video,
        output_dir=output_dir,
        level_name="video",
        val_scores=val_scores_video,

    )

    # ── 6. OCSVM — sequence level ─────────────────────────────
    ocsvm_nu: float = 0.001
    print("\n── Fitting OCSVM (sequence level) ──────────────────")
    seq_ocsvm_dir = Path(output_dir) / "per_sequence" / "ocsvm"
    seq_out = fit_ocsvm_and_score(
        id_scores=val_scores,          # fit on validation (in-distribution)
        test_scores_map=all_scores,
        val_scores=val_scores,
        level_name="sequence",
        out_dir=seq_ocsvm_dir,
        nu=ocsvm_nu,
        kernel="rbf",
        gamma="scale",
    )

    plot_ocsvm_for_all_frauds(
        ocsvm_results=seq_out,
        id_scores=val_scores,
        test_scores_map=all_scores,
        level_name="sequence",
        out_dir=seq_ocsvm_dir,
    )

    print(f"  OCSVM sequence  nu={seq_out['nu']}  kernel={seq_out['kernel']}")
    print(f"  Val OOD rate (boundary): {seq_out['thresholds']['val_ood_rate']:.2%}")
    print(f"  Thresholds (val-calibrated): "
        f"p95={seq_out['thresholds']['p95']:.4f}  "
        f"p99={seq_out['thresholds']['p99']:.4f}")
    print("  Test AUROCs (sequence):")
    for k, v in list(seq_out["auroc"].items()):
        ood_p95 = seq_out["ood_rates"][k]["p95"]
        ood_p99 = seq_out["ood_rates"][k]["p99"]
        print(f"    {k:30s}  AUROC={v:.4f}  "
            f"OOD@p95={ood_p95:.2%}  OOD@p99={ood_p99:.2%}")
    # ── 7. OCSVM — video level ────────────────────────────────
    print("\n── Fitting OCSVM (video level) ──────────────────────")
    vid_ocsvm_dir = Path(output_dir) / "per_video" / "ocsvm"
    vid_out = fit_ocsvm_and_score(
        id_scores=val_scores_video,
        test_scores_map=all_video_scores,
        val_scores=val_scores_video,
        level_name="video",
        out_dir=vid_ocsvm_dir,
        nu=ocsvm_nu,
        kernel="rbf",
        gamma=0.1, #"scale",
    )

    plot_ocsvm_for_all_frauds(
        ocsvm_results=vid_out,
        id_scores=val_scores_video,
        test_scores_map=all_video_scores,
        level_name="video",
        out_dir=vid_ocsvm_dir,
    )


    print(f"  OCSVM video  nu={vid_out['nu']}  kernel={vid_out['kernel']}")
    print(f"  Val OOD rate (boundary): {vid_out['thresholds']['val_ood_rate']:.2%}")
    print(f"  Thresholds (val-calibrated): "
        f"p95={vid_out['thresholds']['p95']:.4f}  "
        f"p99={vid_out['thresholds']['p99']:.4f}")
    print("  Test AUROCs (video):")
    for k, v in list(vid_out["auroc"].items()):
        ood_p95 = vid_out["ood_rates"][k]["p95"]
        ood_p99 = vid_out["ood_rates"][k]["p99"]
        print(f"    {k:30s}  AUROC={v:.4f}  "
            f"OOD@p95={ood_p95:.2%}  OOD@p99={ood_p99:.2%}")
    
    # ── 8. Grouped analysis ───────────────────────────────────
    print("\n── Grouped analysis (sequence level) ───────────────")
    seq_stats, grouped_seq_scores = run_grouped_analysis_ocsvm(
        all_scores=all_scores,
        ocsvm_results=seq_out,
        val_thresholds=val_thresholds,
        output_dir=output_dir,
        level_name="sequence",
    )

    plot_ocsvm_val_vs_test(
        ocsvm_results=seq_out,
        id_scores=val_scores,
        origins_test_scores=all_scores["origins"],
        grouped_scores={k: v for k, v in grouped_seq_scores.items()
                        if k != "origins"},
        level_name="sequence",
        out_dir=Path(output_dir) / "per_sequence" / "ocsvm",
    )

    print("\n── Grouped analysis (video level) ───────────────────")
    vid_stats, grouped_vid_scores = run_grouped_analysis_ocsvm(
        all_scores=all_video_scores,
        ocsvm_results=vid_out,
        val_thresholds=val_thresholds_video,
        output_dir=output_dir,
        level_name="video",
    )

    plot_ocsvm_val_vs_test(
        ocsvm_results=vid_out,
        id_scores=val_scores_video,
        origins_test_scores=all_video_scores["origins"],
        grouped_scores={k: v for k, v in grouped_vid_scores.items()
                        if k != "origins"},
        level_name="video",
        out_dir=Path(output_dir) / "per_video" / "ocsvm",
    )

    return {
        "val_thresholds": val_thresholds,
        "val_thresholds_video": val_thresholds_video,
        "results_per_sequence": all_results,
        "results_per_video": all_video_results,
        "video_ids": {name: ids for name, ids in all_video_ids.items()},
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate hologram fraud detection")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON results to this path")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Directory for SVG plots (default: <log_dir>/plots)")
    args = parser.parse_args()
    print(args)

    # ── config ────────────────────────────────────────────────
    config = Config.from_yaml(args.config)
    config.device = args.device
    device = torch.device(args.device)

    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    # ── model ─────────────────────────────────────────────────
    model = create_model(
        model_type=config.model.type,
        feature_dim=config.model.feature_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        max_seq_len=config.model.max_seq_len,
    )

    ckpt_path = args.checkpoint or (Path(config.paths.log_dir) / config.run_name / "best_model.ckpt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.model.type}  |  params: {n_params:,}  |  VAE: {is_vae(model)}")

    # ── data ──────────────────────────────────────────────────
    data_path = Path(config.paths.data_dir) / config.run_name / "full_data.pt"
    print(f"Loading data: {data_path}")
    sequences, sequences_fakeholo, splits = load_data(data_path)
    print(f"Loaded {len(sequences)} sequence groups, "
          f"{len(sequences_fakeholo)} fraud types")

    (
        _,
        val_loader,
        test_loader_origins,
        test_loaders_fake,
        val_video_names,
        test_video_names_origins,
        test_video_names_fake,
    ) = get_dataloaders(sequences, sequences_fakeholo, splits)

    # ── fraud names ───────────────────────────────────────────
    fake_holo_names = [
        "origins",
        "photo_holo_copy_ID",
        "photo_holo_copy_passport",
        "copy_without_holo_ID",
        "copy_without_holo_passport",
        "pseudo_holo_copy_ID",
        "pseudo_holo_copy_passport",
        "photo_replacement_ID",
        "photo_replacement_passport",
        "plastified_lowreflect_ID",
        "plastified_noholo_ID",
        "no_holo_ID",
        "no_holo_passport",
        "swap_ID",
        "swap_passport",
        "swap_three_ID",
        "plain_holo_ID",
        "plain_holo_passport",
        "leaf_holo_ID",
        "leaf_holo_passport",
        "double_sticker_ID",
        "holo_completemask_ID",
        "holo_star_world_ID",
        "laser_ID",
        "plastified_led_ID"
    ]

    all_test_loaders = [test_loader_origins] + test_loaders_fake
    all_test_video_names = [test_video_names_origins] + test_video_names_fake
    names = fake_holo_names[: len(all_test_loaders)]

    # ── resolve plot directory ────────────────────────────────
    plot_dir = args.plot_dir or (Path(config.paths.log_dir) / config.run_name / "plots")

    # ── run evaluation ────────────────────────────────────────
    results = test(
        model=model,
        val_dataloader=val_loader,
        val_video_names=val_video_names,
        test_dataloaders=all_test_loaders,
        test_video_names_list=all_test_video_names,
        fake_holo_names=names,
        device=device,
        output_dir=plot_dir,
    )

    # ── save JSON ─────────────────────────────────────────────
    out_path = args.output or (Path(config.paths.log_dir) / config.run_name / "test_results.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved → {out_path}")
    print(f"Plots saved  → {plot_dir}/")


if __name__ == "__main__":
    main()
