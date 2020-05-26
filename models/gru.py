import torch
import numpy as np
from torch import nn


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
            batch_first=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=bidirectional, num_layers=num_layers, batch_first=batch_first)
        if dropout:
            assert np.logical_and(dropout > 0, dropout < 1), "Dropout must be between (0, 1)."
            self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """Process data."""
        hidden = self.init_hidden(input)
        output, hidden = self.gru(input, hidden)
        output = self.out(output)
        return output, hidden

    def init_hidden(self, input):
        """Initialize hidden states."""
        multiplier = 1
        assert self.batch_first, "Only designed for batch_first right now."
        if self.bidirectional:
            multiplier = 2
        dim_zero = self.num_layers * multiplier
        return torch.zeros(
            dim_zero,
            input.size(0), self.hidden_size, device=input.device)

