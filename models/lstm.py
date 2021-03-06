import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class LSTM(nn.Module):
    """LSTM with a linear readout shared across timesteps."""
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
        super(LSTM, self).__init__()
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
            
        self.lstm = nn.LSTM(
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
        hidden, cell = self.init_hidden(input)
        output, hidden, cell = self.lstm(input, hidden, cell)
        output = self.out(output)
        return output, hidden

    def init_hidden(self, input):
        """Initialize hidden states."""
        assert self.batch_first, "Only designed for batch_first right now."
        dim_zero = self.num_layers * self.multiplier
        h_0 = torch.zeros(
            dim_zero,
            input.size(0), self.hidden_size, device=input.device)
        c_0 = torch.zeros_like(h_0)
        return h_0, c_0

