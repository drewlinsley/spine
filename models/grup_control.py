import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class GRUP_CONTROL(nn.Module):
    """GRUP CONTROL with a linear readout shared across timesteps."""
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            bptt,
            attention=False,
            emb_size=2,
            dropout=False,
            bidirectional=False,
            num_layers=1,
            alpha=1.,
            max_length=50,
            batch_first=False):
        super(GRUP_CONTROL, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.attention = attention
        self.multiplier = 1
        self.alpha = alpha
        if self.bidirectional:
            self.multiplier = 2
        if self.attention:
            self.attn = nn.Linear(self.hidden_size, max_length)
            self.attn_combine = nn.Linear(self.hidden_size, self.hidden_size)
            
        self.input = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=4,
            stride=1,
            padding=3,
            dilation=2)
        self.gru_enc = nn.GRU(
            hidden_size,
            emb_size * 2,  # mu/sigma
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first)
        self.gru_dec = nn.GRU(
            emb_size * 2,
            hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first)
        # self.out = nn.Linear(self.multiplier * output_size, output_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """Process data."""
        if self.attention:
            raise RuntimeError("Attention is not working properly.")
            attn_weights = F.softmax(self.attn(input))
            input = torch.bmm(
                attn_weights.unsqueeze(0),
                input.unsqueeze(0))
        input = self.input(input.permute(0, 2, 1))
        input = F.elu(input.permute(0, 2, 1))
        hidden_enc = self.init_hidden(input, self.gru_enc.hidden_size)
        hidden_dec = self.init_hidden(input, self.gru_dec.hidden_size)
        output, hidden = self.gru_enc(input, hidden_enc)
        output, hidden = self.gru_dec(output, hidden_dec)
        output = self.out(output)
        return output, hidden

    def init_hidden(self, d1, d2):
        """Initialize hidden states."""
        if self.batch_first:
            dim_zero = self.num_layers * self.multiplier
            return torch.zeros(
                dim_zero, d1.size(0), d2, device=d1.device)
        else:
            dim_zero = self.num_layers * self.multiplier
            return torch.zeros(
                dim_zero, d1.size(1), d2, device=d1.device)

