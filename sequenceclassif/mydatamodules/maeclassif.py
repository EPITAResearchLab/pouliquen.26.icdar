from transformers import VideoMAEConfig, VideoMAEForVideoClassification
from transformers import get_cosine_schedule_with_warmup
import lightning as pl
import torch
from torch import nn
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import numpy as np

class OVDDetectionModelMAE(pl.LightningModule):
    def __init__(self, lr=1e-4, warmup_ratio=0.1):
        super(OVDDetectionModelMAE, self).__init__()
        #self.lr = lr
        self.save_hyperparameters("lr", "warmup_ratio")

        # MobileViT encoder from timm
        config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-small-finetuned-ssv2")
        config.num_classes = 2
        config.num_frames = 5

        self.model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-small-finetuned-ssv2",
            ignore_mismatched_sizes=True,
            config=config
        )

        self.model.classifier = torch.nn.LazyLinear(2)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        #x, y = batch
        x = batch["video_tensor"]
        y = batch["label"]
        logits = self(x)
        
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_video_ids = []
        self._val_seq_accs = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # x, y = batch
        x = batch["video_tensor"]
        y = batch["label"]
        logits = self(x)
        
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)     
        # if dataloader_idx == 0:
        #     self.log('val_loss_onlyreal', loss, prog_bar=True)
        #     self.log('val_acc_onlyreal', acc, prog_bar=True)            
        # elif dataloader_idx == 1:
        #     self.log('val_loss', loss, prog_bar=True)
        # elif dataloader_idx == 2:
        #     self.log('val_loss_onlysynthetic', loss, prog_bar=True)
        #     self.log('val_acc_onlysynthetic', acc, prog_bar=True)
        # elif dataloader_idx == 3:
        #     self.log("val_loss", loss, prog_bar=True)
        #     self.log("val_acc", acc, prog_bar=True)
        #     self._val_video_ids.append(batch["video_id"])
        #     self._val_seq_accs.append((preds == y).float())

        #self.log('val_loss', loss, prog_bar=True)
        #self.log('val_acc', acc, prog_bar=True)

        if batch_idx == 0:
            self._log_sequences_with_labels(x, y, preds, max_samples=4)
        
    def on_validation_epoch_end(self):
        if self._val_seq_accs:
            video_ids = torch.cat(self._val_video_ids)       # Shape: [N]
            seq_accs = torch.cat(self._val_seq_accs)         # Shape: [N]
        
            # Obtain unique video IDs and inverse indices
            unique_vids, inverse_indices = torch.unique(video_ids, return_inverse=True)
            num_videos = unique_vids.size(0)
        
            # Initialize tensors to accumulate sums and counts
            acc_sums = torch.zeros(num_videos, dtype=torch.float, device=seq_accs.device)
            counts = torch.zeros(num_videos, dtype=torch.float, device=seq_accs.device)
        
            # Aggregate sums using scatter_add_
            acc_sums.scatter_add_(0, inverse_indices, seq_accs)
            counts.scatter_add_(0, inverse_indices, torch.ones_like(seq_accs))
        
            # Compute average accuracy per video
            avg_accs = acc_sums / counts
        
            # Compute overall video-level accuracy
            video_accuracy = avg_accs.mean()
        
            # Log the video-level accuracy
            self.log("val_video_acc", video_accuracy, prog_bar=True)

            """
            # Simple vectorized implementation using unique
            unique_ids = torch.unique(video_ids)
            
            # Calculate and log mean accuracy per video
            video_accs = []
            for vid in unique_ids:
                mask = (video_ids == vid)
                acc = seq_accs[mask].mean().item()
                video_accs.append(acc)
                # self.log(f'video_{vid.item()}_acc', acc)
            
            # Log mean accuracy across all videos (each video counts equally)
            mean_video_acc = sum(video_accs) / len(video_accs)
            self.log('mean_video_acc', mean_video_acc, prog_bar=True)
            """
        # add the log of per type of fraud ....

    def configure_optimizers(self):
        # 1) create optimizer
        #param_groups = _get_param_groups(self.model, weight_decay=0.05)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        # 2) compute total steps automatically
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_ratio * total_steps)

        # 3) create scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # 4) return in Lightning‐friendly format
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }
    
    def _log_sequences_with_labels(self, x, y, preds, max_samples=4):
        """
        Create a matplotlib figure with sequences arranged in rows with clear labels
        """
        batch_size = x.size(0)
        num_samples = min(batch_size, max_samples)
        
        class_names = {0: "Real", 1: "Fake"}
        
        # Create figure with subplots
        fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Video Sequences - Epoch {self.current_epoch}', fontsize=16)
        
        for i in range(num_samples):
            sequence = x[i]  # [frames, channels, height, width]
            true_label = y[i].item()
            pred_label = preds[i].item()
            
            # Convert to numpy and normalize
            seq_np = sequence.cpu().numpy()
            if seq_np.min() < 0 or seq_np.max() > 1:
                seq_np = (seq_np - seq_np.min()) / (seq_np.max() - seq_np.min() + 1e-8)
            seq_np = np.clip(seq_np, 0, 1)
            
            # Plot each frame
            for frame_idx in range(5):
                ax = axes[i, frame_idx]
                
                # Convert from CHW to HWC for matplotlib
                frame = seq_np[frame_idx].transpose(1, 2, 0)
                ax.imshow(frame)
                ax.axis('off')
                
                # Add frame number
                ax.set_title(f'Frame {frame_idx + 1}', fontsize=10)
            
            # Add sequence label on the left
            is_correct = "✓" if true_label == pred_label else "✗"
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i, 0].text(-0.1, 0.5, f'Seq {i}\n{is_correct}\nTrue: {class_names[true_label]}\nPred: {class_names[pred_label]}', 
                           transform=axes[i, 0].transAxes, 
                           fontsize=12, 
                           verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        
        plt.tight_layout()
        
        # Convert matplotlib figure to tensor for TensorBoard
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to PIL Image then to tensor
        from PIL import Image
        img = Image.open(buf)
        img_array = np.array(img)
        buf.close()
        plt.close(fig)
        
        # Convert to tensor [C, H, W]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        
        # Log to TensorBoard
        self.logger.experiment.add_image(
            f"sequences_detailed/epoch_{self.current_epoch}",
            img_tensor,
            global_step=self.current_epoch
        )
