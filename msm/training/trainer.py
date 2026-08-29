"""Main trainer class for hologram fraud detection."""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from configs.config import Config
from losses.losses import CombinedVAELoss
from .early_stopping import EarlyStopping


class Trainer:
    """
    Unified trainer for all model types.
    
    Features:
        - Multi-dataset validation
        - Gradient clipping and NaN detection
        - TensorBoard logging
        - Early stopping
        - Checkpoint saving
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loaders: Dict[str, DataLoader],
        masking_fn,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.masking_fn = masking_fn
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.training.lr
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.training.num_epochs - 2
        )
        
        # Loss function
        self.criterion = CombinedVAELoss(
            alpha=config.loss.alpha,
            beta=config.loss.beta,
            gamma=config.loss.gamma,
            kl_warmup_epochs=config.loss.kl_warmup,
            eps=float(config.loss.eps)
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta
        )
        
        # Logging
        self.writer = SummaryWriter(log_dir=config.paths.log_dir + "/" + config.run_name)
        
        # Tracking
        self.best_model = None
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.history = []
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.criterion.set_epoch(epoch)
        
        total_loss = 0.0
        loss_components = {"mse": 0.0, "cosine": 0.0, "kl": 0.0}
        num_batches = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            # Random temporal flip
            if torch.rand(1).item() < 0.5:
                batch = batch.flip(dims=[1])
            
            # Apply masking
            masked_batch, mask = self.masking_fn(batch)
            # Forward pass
            # if hasattr(self.model, 'forward') and 'vae' in str(type(self.model)):
            if hasattr(self.model, 'compute_kl_divergence'):
                output, mu, logvar = self.model(masked_batch, mask)
                loss, loss_dict = self.criterion(output, batch, mask, mu, logvar)
            else:
                output = self.model(masked_batch)
                loss, loss_dict = self.criterion(output, batch, mask)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"WWWWWWarninggggggggggg: NaN loss detected, skipping batch")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.max_grad_norm
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k in loss_components:
                    loss_components[k] += v
            num_batches += 1
        
        self.scheduler.step()
        
        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        for k in loss_components:
            loss_components[k] /= max(num_batches, 1)
        
        return {"loss": avg_loss, **loss_components}
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, List[float]]:
        """Run validation on a single dataloader."""
        self.model.eval()
        losses = []
        
        for batch in loader:
            batch = batch.to(self.device)
            masked_batch, mask = self.masking_fn(batch)
            
            if hasattr(self.model, 'forward'):
                try:
                    output, mu, logvar = self.model(masked_batch, mask)
                    loss, _ = self.criterion(output, batch, mask, mu, logvar)
                except:
                    output = self.model(masked_batch)
                    loss, _ = self.criterion(output, batch, mask)
            
            losses.append(loss.item())
        
        return np.mean(losses), losses
    
    def train(self) -> Dict:
        """Run full training loop."""
        print(f"Starting training for {self.config.training.num_epochs} epochs")
        
        for epoch in range(self.config.training.num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            
            # Validation on all datasets
            val_metrics = {}
            for name, loader in self.val_loaders.items():
                val_loss, _ = self.validate(loader)
                val_metrics[name] = val_loss
                self.writer.add_scalar(f"val/{name}", val_loss, epoch)
            
            # Model selection
            primary_val_loss = val_metrics.get("val_data", list(val_metrics.values())[0])
            combined_loss = primary_val_loss + abs(primary_val_loss - train_metrics["loss"])
            
            if combined_loss < self.best_loss:
                self.best_loss = combined_loss
                self.best_epoch = epoch + 1
                self.best_model = copy.deepcopy(self.model)
                print(f"  New best model at epoch {epoch+1}")
            
            # Early stopping
            if self.early_stopping(primary_val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Logging
            print(f"Epoch {epoch+1}: train_loss={train_metrics['loss']:.4f}, "
                  f"val_loss={primary_val_loss:.4f}")
        
        self.writer.close()
        
        # Restore best model
        if self.best_model is not None:
            self.model = self.best_model
        
        return {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "final_val_metrics": val_metrics
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss
        }, path)
