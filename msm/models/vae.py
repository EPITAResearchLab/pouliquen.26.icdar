"""Masked Variational Autoencoder for anomaly detection."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .positional import PositionalEncoding


class MaskedVAE(nn.Module):
    """
    Masked Variational Autoencoder with Transformer encoder/decoder.
    
    This model combines masked reconstruction with VAE for robust anomaly
    detection in hologram sequences. The VAE component enables probabilistic
    latent space modeling, providing multiple signals for anomaly scoring.
    
    Architecture:
        1. Encoder: Transformer encoder → attention pooling → μ, σ projection
        2. Reparameterization: z = μ + σ * ε (training), z = μ (inference)
        3. Decoder: Transformer decoder with cross-attention to encoder
    
    Args:
        feature_dim: Input/output feature dimension
        hidden_dim: Transformer feedforward dimension
        latent_dim: VAE latent space dimension
        num_encoder_layers: Number of encoder transformer layers
        num_decoder_layers: Number of decoder transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
    
    Shape:
        - Input: (batch, seq_len, feature_dim)
        - Output: (batch, seq_len, feature_dim), mu, logvar
    """
    
    def __init__(
        self,
        feature_dim: int = 320,
        hidden_dim: int = 200,
        # latent_dim: int = 64,
        latent_dim: int = 5,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 5,
        dropout: float = 0.1,
        max_seq_len: int = 60,
        batch_first: bool = True,
    ):
        super(MaskedVAE, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.batch_first = batch_first
        
        # Learnable mask token for replacing masked positions
        self.mask_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            emb_size=feature_dim,
            dropout=dropout,
            maxlen=max_seq_len,
            batch_first=self.batch_first
        )
        
        # ========== Encoder ==========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="gelu",
            dropout=dropout,
            batch_first=self.batch_first
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Attention pooling for sequence aggregation
        self.attention_pool = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # VAE projection heads
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
        
        # ========== Decoder ==========
        # Project latent back to feature dim
        self.latent_to_decoder = nn.Linear(latent_dim, feature_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="gelu",
            dropout=dropout,
            batch_first=self.batch_first
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output projection
        self.output_layer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Layer norm for encoder output
        self.encoder_norm = nn.LayerNorm(feature_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input sequence to latent distribution parameters.
        
        Args:
            x: Input tensor (batch, seq_len, feature_dim)
            mask: Optional mask for positions (batch, seq_len), True = masked
        
        Returns:
            encoder_output: Transformer encoder output
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Apply mask token to masked positions
        if mask is not None:
            batch_size = x.size(0)
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            mask_tokens = self.mask_token.expand(batch_size, x.size(1), -1)
            x = torch.where(mask_expanded, mask_tokens, x)
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoder
        encoder_output = self.encoder(x)
        encoder_output = self.encoder_norm(encoder_output)
        
        # Attention pooling for latent projection
        attention_weights = self.attention_pool(encoder_output)  # (batch, seq_len, 1)
        pooled = (encoder_output * attention_weights).sum(dim=1)  # (batch, feature_dim)
        
        # VAE projections
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        
        return encoder_output, mu, logvar
    
    def reparameterize(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            training: If True, sample from distribution; else use mean
        
        Returns:
            Sampled latent vector
        """
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(
        self,
        z: torch.Tensor,
        encoder_output: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """
        Decode latent vector to reconstructed sequence.
        
        Args:
            z: Latent vector (batch, latent_dim)
            encoder_output: Encoder output for cross-attention
            seq_len: Target sequence length
        
        Returns:
            Reconstructed sequence (batch, seq_len, feature_dim)
        """
        batch_size = z.size(0)
        
        # Project latent to decoder input
        z_proj = self.latent_to_decoder(z)  # (batch, feature_dim)
        
        # Expand to sequence and add positional encoding
        decoder_input = z_proj.unsqueeze(1).expand(-1, seq_len, -1)
        decoder_input = self.positional_encoding(decoder_input)
        
        # Transformer decoder with cross-attention
        decoder_output = self.decoder(
            decoder_input,
            encoder_output
        )
        
        # Output projection
        output = self.output_layer(decoder_output)
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor (batch, seq_len, feature_dim)
            mask: Optional mask for positions (batch, seq_len), True = masked
        
        Returns:
            output: Reconstructed sequence
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        seq_len = x.size(1)
        
        # Encode
        encoder_output, mu, logvar = self.encode(x, mask)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar, self.training)
        
        # Decode
        output = self.decode(z, encoder_output, seq_len)
        
        return output, mu, logvar
    
    def get_latent(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get latent representation without decoding.
        
        Useful for anomaly scoring based on latent space.
        
        Args:
            x: Input tensor (batch, seq_len, feature_dim)
            mask: Optional mask for positions
        
        Returns:
            Latent vector (batch, latent_dim)
        """
        _, mu, logvar = self.encode(x, mask)
        return self.reparameterize(mu, logvar, training=False)
    
    def compute_kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence from standard normal prior.
        
        KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            KL divergence (scalar)
        """
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
