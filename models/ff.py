import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class FF(nn.Module):
    """FF control model with a linear readout shared across timesteps."""
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            bptt,
            attention=False,
            dropout=False,
            bidirectional=False,
            num_layers=1,
            kernel_size=2,
            max_length=50,
            batch_first=False):
        super(FF, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.attention = attention
        self.layers = []
        self.layers.append(
            nn.Sequential(
                nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size, stride=1, padding=1),
                nn.ReLU()))
        for l in range(num_layers):
            self.layers.append(
                nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, stride=1, padding=1),
                nn.ReLU()))
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """Process data."""
        for layer in self.layers:
            input = layer(input)
        output = self.out(input)
        return output, None

