import torch
import numpy as np
from torch import nn


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
            batch_first=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        import ipdb;ipdb.set_trace()
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=bidirectional, num_layers=num_layers, batch_first=batch_first)
        if dropout:
            assert np.logical_and(dropout > 0, dropout < 1), "Dropout must be between (0, 1)."
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        """Process data."""
        hidden = self.init_hidden()
        output, hidden = self.gru(input, hidden)
        output = self.out(output)
        return output, hidden

    def init_hidden(self):
        """Initialize hidden states."""
        dim_zero = 1
        if self.bidirectional:
            dim_zero = 2
        return torch.zeros(dim_zero, 1, self.hidden_size, device=device)

