import torch
import numpy as np
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from models.vae import MaskedVAE

def is_vae(model: torch.nn.Module) -> bool:
    return isinstance(model, MaskedVAE)

@torch.no_grad()
def compute_anomaly_scores(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Return a dict of 1-D numpy arrays (one score per sample).

    Signals for every model:
        recon_mse       : mean MSE between reconstructed and original frames
        recon_cosine    : mean cosine distance between reconstructed and original frames
        max_frame_error : max single-frame MSE across sequence
        seq_mse         : mean pairwise MSE between ALL frames in the original sequence
        seq_cosine      : mean pairwise cosine distance between ALL frames in the original sequence

    Additional for VAE:
        kl_divergence, elbo
    
    The seq_mse and seq_cosine metrics capture the overall frame-to-frame variability 
    within each sequence (all pairs), allowing comparison with reconstruction error.
    """
    model.eval()
    vae = is_vae(model)

    recon_mse_all = []
    recon_cos_all = []
    max_frame_all = []
    kl_all = []
    
    # New: all-pairs frame-to-frame metrics within each sequence
    seq_mse_all = []
    seq_cos_all = []

    for batch in dataloader:
        batch = batch.to(device)
        # batch shape: (B, T, D) where B=batch, T=frames, D=features
        B, T, D = batch.shape

        if vae:
            output, mu, logvar = model(batch)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
            kl_all.append(kl.cpu())
        else:
            output = model(batch)

        # ── Reconstruction metrics (output vs batch) ──────────────
        per_frame_mse = ((output - batch) ** 2).mean(dim=-1)  # (B, T)
        recon_mse_all.append(per_frame_mse.mean(dim=-1).cpu())  # mean over frames
        max_frame_all.append(per_frame_mse.max(dim=-1).values.cpu())

        cos = 1.0 - F.cosine_similarity(output, batch, dim=-1)  # (B, T)
        recon_cos_all.append(cos.mean(dim=-1).cpu())  # mean over frames

        # ── All-pairs frame-to-frame metrics within original sequence ──
        # Compare every frame to every other frame in the sequence
        
        # Expand for pairwise comparison
        # batch_i: (B, T, 1, D) - each frame
        # batch_j: (B, 1, T, D) - compared against all frames
        batch_i = batch.unsqueeze(2)  # (B, T, 1, D)
        batch_j = batch.unsqueeze(1)  # (B, 1, T, D)
        
        # Pairwise MSE: (B, T, T)
        pairwise_mse = ((batch_i - batch_j) ** 2).mean(dim=-1)  # (B, T, T)
        
        # Pairwise cosine distance: (B, T, T)
        # Normalize for efficient cosine computation via matmul
        batch_norm = F.normalize(batch, p=2, dim=-1)  # (B, T, D)
        cos_sim = torch.bmm(batch_norm, batch_norm.transpose(1, 2))  # (B, T, T)
        pairwise_cos = 1.0 - cos_sim  # (B, T, T)
        
        # Mask to exclude diagonal (frame compared to itself)
        mask = ~torch.eye(T, dtype=torch.bool, device=device)  # (T, T)
        
        # Mean over all pairs excluding self-comparisons
        # Number of valid pairs per sample: T * (T - 1)
        seq_mse_all.append(pairwise_mse[:, mask].view(B, -1).mean(dim=-1).cpu())
        seq_cos_all.append(pairwise_cos[:, mask].view(B, -1).mean(dim=-1).cpu())

    scores = {
        "recon_mse": torch.cat(recon_mse_all).numpy(),
        "recon_cosine": torch.cat(recon_cos_all).numpy(),
        "max_frame_error": torch.cat(max_frame_all).numpy(),
        "seq_mse": torch.cat(seq_mse_all).numpy(),
        "seq_cosine": torch.cat(seq_cos_all).numpy(),
    }
    if vae:
        kl_np = torch.cat(kl_all).numpy()
        scores["kl_divergence"] = kl_np
        scores["elbo"] = scores["recon_mse"] + 0.1 * kl_np

    return scores

def aggregate_scores_by_video(
    scores: dict[str, np.ndarray],
    video_names: list[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Aggregate per-sequence scores to per-video scores (mean per video).

    Parameters
    ----------
    scores : dict mapping signal name → 1-D array (one value per sequence)
    video_names : list of video identifiers, one per sequence (parallel to arrays)

    Returns
    -------
    agg_scores : dict mapping signal name → 1-D array (one value per video)
    unique_videos : ordered list of unique video names (parallel to arrays)
    """
    unique_videos = list(dict.fromkeys(video_names))  # preserve order
    video_to_idx = defaultdict(list)
    for i, v in enumerate(video_names):
        video_to_idx[v].append(i)

    agg_scores = {}
    for key, vals in scores.items():
        agg = np.array([vals[video_to_idx[v]].mean() for v in unique_videos])
        # agg = np.array([np.median(vals[video_to_idx[v]]) for v in unique_videos])
        agg_scores[key] = agg

    return agg_scores, unique_videos
