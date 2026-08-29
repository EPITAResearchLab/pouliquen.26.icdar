"""Positional encoding for transformer models."""

import math
import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    
    Adds position-dependent signals to input embeddings to provide
    temporal/sequential information to the model.
    
    Args:
        emb_size: Embedding dimension size
        dropout: Dropout probability
        maxlen: Maximum sequence length supported
    
    Shape:
        - Input: (seq_len, batch, emb_size) or (batch, seq_len, emb_size)
        - Output: Same as input
    """
    
    def __init__(
        self, 
        emb_size: int, 
        dropout: float = 0.1, 
        maxlen: int = 60,
        batch_first: bool = False
    ):
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first
        
        # Compute positional encoding using sine and cosine
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        
        # Add batch dimension for broadcasting
        if batch_first:
            pos_embedding = pos_embedding.unsqueeze(0)  # (1, maxlen, emb_size)
        else:
            pos_embedding = pos_embedding.unsqueeze(1)  # (maxlen, 1, emb_size)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)
    
    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            token_embedding: Input tensor of shape (seq_len, batch, emb_size) 
                           or (batch, seq_len, emb_size) if batch_first=True
        
        Returns:
            Tensor with positional encoding added
        """
        if self.batch_first:
            seq_len = token_embedding.size(1)
            return self.dropout(
                token_embedding + self.pos_embedding[:, :seq_len, :]
            )
        else:
            seq_len = token_embedding.size(0)
            return self.dropout(
                token_embedding + self.pos_embedding[:seq_len, :, :]
            )


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    
    Instead of fixed sinusoidal patterns, learns position embeddings
    during training.
    
    Args:
        emb_size: Embedding dimension size
        dropout: Dropout probability
        maxlen: Maximum sequence length supported
        batch_first: If True, input is (batch, seq, dim), else (seq, batch, dim)
    """
    
    def __init__(
        self, 
        emb_size: int, 
        dropout: float = 0.1, 
        maxlen: int = 60,
        batch_first: bool = False
    ):
        super(LearnablePositionalEncoding, self).__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(maxlen, emb_size) * 0.02)
    
    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        """Add learnable positional encoding to input embeddings."""
        if self.batch_first:
            seq_len = token_embedding.size(1)
            pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(0)
        else:
            seq_len = token_embedding.size(0)
            pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(1)
        
        return self.dropout(token_embedding + pos_emb)
