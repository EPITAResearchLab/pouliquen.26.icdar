"""Models module for hologram fraud detection.

Available models:
    - TransformerEncoder: Transformer-based encoder for long sequences
    - GRUEncoder: Bidirectional GRU for shorter sequences
    - HybridTransformerGRU: Combined transformer + GRU
    - MaskedVAE: VAE with masked reconstruction for anomaly detection
"""

from typing import Literal

from .transformers import ContinuousTransformerModel
from .gru import GRUWithEmbeddings
from .hybrid import HybridTransformerGRU
from .vae import MaskedVAE


def create_model(
    model_type: Literal["transformer", "gru", "hybrid", "vae"],
    feature_dim: int = 320,
    hidden_dim: int = 200,
    latent_dim: int = 64,
    num_layers: int = 4,
    num_heads: int = 5,
    dropout: float = 0.1,
    max_seq_len: int = 60,
    **kwargs
):
    """
    Factory function to create a model based on type.
    
    Args:
        model_type: Type of model to create
        feature_dim: Input/output feature dimension
        hidden_dim: Hidden layer dimension
        latent_dim: Latent dimension (for VAE)
        num_layers: Number of encoder/decoder layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
        **kwargs: Additional model-specific arguments
    
    Returns:
        Instantiated model
    
    Example:
        >>> model = create_model("vae", feature_dim=320, latent_dim=64)
    """
    print("===========", max_seq_len)
    if model_type == "transformer":
        return ContinuousTransformerModel(
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            # dropout=dropout,
            maxlen=max_seq_len,
            **kwargs
        )
    elif model_type == "gru":
        return GRUWithEmbeddings(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
    elif model_type == "hybrid":
        return HybridTransformerGRU(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_transformer_layers=num_layers // 2,
            num_gru_layers=num_layers // 2,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            **kwargs
        )
    elif model_type == "vae":
        return MaskedVAE(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_encoder_layers=num_layers // 2,
            num_decoder_layers=num_layers // 2,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    "PositionalEncoding",
    "LearnablePositionalEncoding",
    "TransformerEncoder",
    "ContinuousTransformerModel",
    "GRUEncoder",
    "GRUWithEmbeddings",
    "HybridTransformerGRU",
    "MaskedVAE",
    "create_model",
]
