import numpy as np
import torch
from torch import nn
from torchvision import transforms as T
from .positional import PositionalEncoding



class ContinuousTransformerModel(nn.Module):
    def __init__(self, feature_dim, num_layers, num_heads, hidden_dim, maxlen):
        super(ContinuousTransformerModel, self).__init__()
        self.positional_encoding = PositionalEncoding(feature_dim, maxlen=maxlen, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(feature_dim, feature_dim)

    def forward(self, src, src_mask=None):
        src = self.positional_encoding(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        return self.output_layer(memory)


# class PositionalEncoding(nn.Module):
#     def __init__(self, emb_size, dropout=0.1, maxlen=60):
#         super(PositionalEncoding, self).__init__()
#         den = torch.exp(-torch.arange(0, emb_size, 2) * np.log(10000) / emb_size)
#         pos = torch.arange(0, maxlen).reshape(maxlen, 1)
#         pos_embedding = torch.zeros((maxlen, emb_size))
#         pos_embedding[:, 0::2] = torch.sin(pos * den)
#         pos_embedding[:, 1::2] = torch.cos(pos * den)
#         pos_embedding = pos_embedding.unsqueeze(-2)

#         self.dropout = nn.Dropout(dropout)
#         self.register_buffer("pos_embedding", pos_embedding)

#     def forward(self, token_embedding):
#         print(f"{token_embedding.shape=}")
#         print(f"{self.pos_embedding.shape=}")

#         return self.dropout(
#             token_embedding + self.pos_embedding[: token_embedding.size(0), :]
#         )