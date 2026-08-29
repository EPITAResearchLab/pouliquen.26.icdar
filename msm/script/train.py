"""Main training script."""
import argparse
import torch
from pathlib import Path

from configs.config import Config
from models.models_init import create_model
from training.trainer import Trainer
from mydatasets.utils import bert_mask
import pickle





def load_data(data_path):
    # data = torch.load(data_path, weights_only=False)
    data = pickle.load(Path(data_path).open("rb"))

    return data

def collate_fn(batch):
    # `batch` will be a list of tensors with shape (S, E)
    # Stack them into a tensor of shape (N, S, E)
    batch = torch.stack(batch, dim=0)  # (N, S, E)
    # Transpose to get (S, N, E)
    batch = batch.transpose(0, 1)  # Now it's (S, N, E)
    return batch

def get_dataloaders(sequences, sequences_fakeholo, splits):
    split_train, split_val, split_test = splits

    # TRAIN
    train_data = torch.cat([sequences[k] for k in split_train if k in sequences], dim=0)
    print(f"{train_data.shape}")
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True#, collate_fn=collate_fn
    )

    # VAL

    val_data = torch.cat([sequences[k] for k in split_val if k in sequences], dim=0)
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=32, shuffle=False#, collate_fn=collate_fn
    )

    # TEST

    test_data = torch.cat([sequences[k] for k in split_test if k in sequences], dim=0)
    test_dataloader_holo = torch.utils.data.DataLoader(
        test_data, batch_size=32, shuffle=False#, collate_fn=collate_fn
    )
    test_dataloaders = []
    for seq_fake in sequences_fakeholo:
        test_data = torch.cat([seq_fake[k] for k in split_test if k in seq_fake], dim=0)
        test_dataloaders.append(
            torch.utils.data.DataLoader(
                test_data, batch_size=32, shuffle=False#, collate_fn=collate_fn
            )
        )
    return (
        train_dataloader,
        val_dataloader,
        test_dataloader_holo,
        test_dataloaders,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    print(args)
    
    # Load config
    config = Config.from_yaml(args.config)
    config.device = args.device
    
    # Set seed
    torch.manual_seed(config.training.seed)
    
    # Create model
    model = create_model(
        model_type=config.model.type,
        feature_dim=config.model.feature_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        max_seq_len=config.model.max_seq_len
    )
    
    print(f"Created {config.model.type} model with {sum(p.numel() for p in model.parameters())} parameters")

    # Load data
    print("Loading data from", Path(config.paths.data_dir) / config.run_name / "full_data.pt")
    sequences, sequences_fakeholo, splits = load_data(Path(config.paths.data_dir) / config.run_name / "full_data.pt")
    print(f"Loaded {len(sequences)} sequence groups")

    dataloaders = get_dataloaders(sequences, sequences_fakeholo, splits)
    
    (
        train_dataloader,
        val_dataloader,
        test_dataloader_noholo, # not used
        test_dataloaders, # not used
    ) = dataloaders
    
    val_loaders = {
        "normal": val_dataloader,
    }

    trainer = Trainer(model,
                      config,
                      train_loader=train_dataloader,
                      val_loaders=val_loaders,
                    #   masking_fn=seq_mask,
                      masking_fn=bert_mask,
                      device=config.device)
    
    trainer.train()
    trainer.save_checkpoint(Path(config.paths.log_dir) / config.run_name / "best_model.ckpt")
    

    
if __name__ == "__main__":
    main()
