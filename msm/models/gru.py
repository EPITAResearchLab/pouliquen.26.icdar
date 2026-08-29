
import torch
import torch.nn as nn
from typing import Optional


class GRUWithEmbeddings(nn.Module):
    def __init__(self, emb_size, hidden_dim, num_layers=2, dropout=0.1, bidirectional=True):
        super(GRUWithEmbeddings, self).__init__()
        print("gruuuuu")

        # Define GRU layer that accepts embedded vectors as input
        self.gru = nn.GRU(input_size=emb_size, hidden_size=hidden_dim, num_layers=num_layers, 
                          batch_first=False, dropout=dropout, bidirectional=bidirectional)

        # Output layer to transform hidden state output into desired output size
        self.output_layer = nn.Linear(hidden_dim * (2 if bidirectional else 1), emb_size)

    def forward(self, x):
        # Pass through GRU
        x, _ = self.gru(x)  # Shape: (batch_size, seq_len, hidden_dim * (2 if bidirectional else 1))

        # Output transformation
        output = self.output_layer(x)  # Shape: (batch_size, seq_len, emb_size)
        return output
