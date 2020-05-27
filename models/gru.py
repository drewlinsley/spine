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
            max_length=50,
            batch_first=True):
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
            self.attn = nn.Linear(self.hidden_size, max_length)
            self.attn_combine = nn.Linear(self.hidden_size, self.hidden_size)
            
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
            raise RuntimeError("Attention is not working properly.")
            attn_weights = F.softmax(self.attn(input))
            input = torch.bmm(
                attn_weights.unsqueeze(0),
                input.unsqueeze(0))
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

