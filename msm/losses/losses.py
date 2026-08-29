"""Combined loss functions for masked reconstruction with VAE."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class CombinedVAELoss(nn.Module):
    """
    Combined loss for VAE with masked reconstruction.
    
    L_total = α * L_mse + β * L_cosine + γ * L_kl
    
    Includes KL annealing to prevent posterior collapse.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
        kl_warmup_epochs: int = 50,
        eps: float = 1e-8
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kl_warmup_epochs = kl_warmup_epochs
        self.eps = eps
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for KL annealing."""
        self.current_epoch = epoch
    
    def get_kl_weight(self) -> float:
        """Get current KL weight with linear warmup."""
        if self.kl_warmup_epochs <= 0:
            return self.gamma
        return min(1.0, self.current_epoch / self.kl_warmup_epochs) * self.gamma
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model output (batch, seq_len, dim)
            targets: Ground truth (batch, seq_len, dim)
            mask: Binary mask (batch, seq_len), 1 = compute loss
            mu: VAE mean (optional)
            logvar: VAE log variance (optional)
        
        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dictionary with individual loss components
        """
        # Ensure mask has correct shape
        # if mask.dim() == 2:
        #     mask = mask.unsqueeze(-1)
        # else:
        #     mask = mask
        
        # Count valid elements
        num_valid = mask.sum()
        if num_valid == 0:
            zero_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            return zero_loss, {"total": 0.0, "mse": 0.0, "cosine": 0.0, "kl": 0.0}
        
        # ============ MSE Loss ============
        mse_per_elem = ((predictions - targets) ** 2).mean(dim=-1)
        mse_loss = (mse_per_elem * mask).sum() / num_valid
        
        # ============ Cosine Loss ============
        pred_norm = predictions / (predictions.norm(dim=-1, keepdim=True) + self.eps)
        tgt_norm = targets / (targets.norm(dim=-1, keepdim=True) + self.eps)
        cosine_sim = (pred_norm * tgt_norm).sum(dim=-1)
        cosine_loss = ((1.0 - cosine_sim) * mask).sum() / num_valid
        
        # ============ KL Loss ============
        kl_loss = torch.tensor(0.0, device=predictions.device)
        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # ============ Combine ============
        kl_weight = self.get_kl_weight()
        total_loss = self.alpha * mse_loss + self.beta * cosine_loss + kl_weight * kl_loss
        
        loss_dict = {
            "total": total_loss.item(),
            "mse": mse_loss.item(),
            "cosine": cosine_loss.item(),
            "kl": kl_loss.item(),
            "kl_weight": kl_weight
        }
        
        return total_loss, loss_dict


def masked_cosine_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Standalone combined cosine + MSE loss function.
    
    For use without VAE component.
    """
    num_valid = mask.sum()
    if num_valid == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    # MSE
    mse = ((predictions - targets) ** 2).mean(dim=-1)
    masked_mse = (mse * mask).sum() / num_valid
    
    # Cosine
    pred_norm = predictions / (predictions.norm(dim=-1, keepdim=True) + eps)
    tgt_norm = targets / (targets.norm(dim=-1, keepdim=True) + eps)
    cosine_sim = (pred_norm * tgt_norm).sum(dim=-1)
    masked_cosine = ((1.0 - cosine_sim) * mask).sum() / num_valid
    
    return alpha * masked_mse + beta * masked_cosine
