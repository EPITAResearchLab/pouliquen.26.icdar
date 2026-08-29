"""Hybrid Transformer + GRU model for masked reconstruction."""

import torch
import torch.nn as nn
from typing import Optional

from .positional import PositionalEncoding


class HybridTransformerGRU(nn.Module):
    """
    Hybrid model combining Transformer and GRU.
    
    Uses Transformer for global attention patterns and GRU for local
    sequential dependencies. This combination can capture both long-range
    relationships and smooth temporal transitions.
    
    Args:
        feature_dim: Input/output feature dimension
        hidden_dim: Hidden dimension for both transformer and GRU
        num_transformer_layers: Number of transformer encoder layers
        num_gru_layers: Number of GRU layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
    
    Shape:
        - Input: (batch, seq_len, feature_dim)
        - Output: Same as input
    """
    
    def __init__(
        self,
        feature_dim: int = 320,
        hidden_dim: int = 200,
        num_transformer_layers: int = 2,
        num_gru_layers: int = 2,
        num_heads: int = 5,
        dropout: float = 0.1,
        max_seq_len: int = 60
    ):
        super(HybridTransformerGRU, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            emb_size=feature_dim,
            dropout=dropout,
            maxlen=max_seq_len,
            batch_first=True
        )
        
        # Transformer encoder for global patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="gelu",
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # GRU for local patterns
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fusion layer to combine transformer and GRU outputs
        gru_output_dim = hidden_dim * 2  # bidirectional
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + gru_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through hybrid model.
        
        Args:
            src: Input tensor of shape (batch, seq_len, feature_dim)
            src_key_padding_mask: Optional mask for padded positions
        
        Returns:
            Reconstructed tensor of shape (batch, seq_len, feature_dim)
        """
        # Add positional encoding
        x = self.positional_encoding(src)
        
        # Transformer branch (global patterns)
        transformer_out = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # GRU branch (local patterns)
        gru_out, _ = self.gru(src)
        
        # Concatenate and fuse
        combined = torch.cat([transformer_out, gru_out], dim=-1)
        fused = self.fusion(combined)
        
        # Add residual and normalize
        output = self.layer_norm(fused + src)
        
        return output
