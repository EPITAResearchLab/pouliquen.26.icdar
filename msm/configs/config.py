"""Configuration management using dataclasses."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal
from pathlib import Path
import json
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    type: Literal["transformer", "gru", "hybrid", "vae"] = "vae"
    feature_dim: int = 320
    hidden_dim: int = 200
    latent_dim: int = 64
    num_layers: int = 4
    num_heads: int = 5
    dropout: float = 0.1
    max_seq_len: int = 60


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    lr: float = 0.0001
    num_epochs: int = 300
    batch_size: int = 16
    seed: int = 42
    max_grad_norm: float = 1.0
    patience: int = 20
    min_delta: float = 0.001


@dataclass
class LossConfig:
    """Loss function configuration."""
    alpha: float = 1.0      # MSE weight
    beta: float = 1.0       # Cosine weight
    gamma: float = 0.1      # KL weight
    kl_warmup: int = 50     # Epochs for KL annealing
    eps: float = 1e-8       # Numerical stability


@dataclass
class DataConfig:
    """Data loading and augmentation configuration."""
    seq_len: int = 60
    sequences_step: int = 20
    min_similarity: float = 0.3
    sample_seq_prob: float = 0.01
    flat_prob: float = 0.02
    validation_batch_size: int = 4


@dataclass
class MaskingConfig:
    """Masking strategy configuration."""
    strategy: Literal["seq_mask", "bert_mask", "modulo_mask"] = "seq_mask"
    mask_prob: float = 0.15
    len_hide: int = 5
    num_masks: int = 2
    noise_std: float = 0.1


@dataclass
class PathConfig:
    """Path configuration."""
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    backbone_path: str = ""
    predata_dir: str = "data/"
    splits_dir: str = "data/splits_kfold_s0/k0/simple/"

@dataclass
class ProjectorConfig:
    backbone_name: str = "mobilevit_xxs"
    model_path: str = ""
    pretrained: bool = True 
    num_classes: int= -1
    dynamic_img_size: bool = False

@dataclass
class Config:
    """
    Main configuration class combining all sub-configurations.
    
    Supports loading from and saving to JSON/YAML files.
    
    Example:
        >>> config = Config.from_yaml("config/default.yaml")
        >>> config.training.lr = 0.001
        >>> config.save_yaml("config/experiment.yaml")
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    device: str = "cuda"
    run_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
            loss=LossConfig(**d.get("loss", {})),
            data=DataConfig(**d.get("data", {})),
            masking=MaskingConfig(**d.get("masking", {})),
            paths=PathConfig(**d.get("paths", {})),
            projector=ProjectorConfig(**d.get("projector", {})),
            device=d.get("device", "cuda"),
            run_name=d.get("run_name", "")
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
    
    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)
    
    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __repr__(self) -> str:
        return f"Config({json.dumps(self.to_dict(), indent=2)})"
