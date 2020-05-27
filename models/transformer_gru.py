import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class GRU(nn.Module):
    """GRU with a linear readout shared across timesteps."""
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            attention=False,
            dropout=False,
            bidirectional=False,
            num_layers=1,
            num_heads=2,
            max_length=50,
            batch_first=False):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.attention = attention
        self.multiplier = 1
        if self.bidirectional:
            self.multiplier = 2
        if self.attention:
            self.attention = nn.Transformer(nheads=num_heads, num_encoder_layers=num_layers)
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first)
        self.out = nn.Linear(self.multiplier * hidden_size, output_size)

    def forward(self, input):
        """Process data."""
        if self.attention:
            pass
        hidden = self.init_hidden(input)
        output, hidden = self.gru(input, hidden)
        output = self.out(output)
        return output, hidden

    def init_hidden(self, input):
        """Initialize hidden states."""
        assert self.batch_first, "Only designed for batch_first right now."
        dim_zero = self.num_layers * self.multiplier
        return torch.zeros(
            dim_zero,
            input.size(0), self.hidden_size, device=input.device)

