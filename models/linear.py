import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
   pad = (kernel_size - 1) * dilation
   return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)


class linear(nn.Module):
    """linear readout across timesteps."""
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
            max_length=50,
            kernel_size=4,
            batch_first=False):
        super(linear, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.attention = attention
        self.bptt = bptt
        self.multiplier = 1
        if self.bidirectional:
            self.multiplier = 2
        if self.attention:
            self.attn = nn.Linear(self.hidden_size, max_length)
            self.attn_combine = nn.Linear(self.hidden_size, self.hidden_size)
        assert kernel_size % 2 == 0, "Needs an even kernel size."
        # self.input = nn.Conv1d(
        #     in_channels=input_size,
        #     out_channels=hidden_size,
        #     kernel_size=kernel_size,
        #     stride=1,
        #     padding=2,
        #     dilation=1)
        self.layers = []
        # self.layers.append(nn.Linear(bptt * input_size, bptt * hidden_size))
        # for _ in range(num_layers - 1):
        #     self.layers.append(nn.Linear(bptt * hidden_size, bptt * hidden_size))
        self.layers.append(CausalConv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=1,
            dilation=2))
        for _ in range(num_layers - 1): 
            self.layers.append(CausalConv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                groups=1,
                dilation=2))
        self.output = nn.Linear(bptt * hidden_size, bptt * output_size)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input):
        """Process data."""
        in_shape = input.shape
        # input = self.input(input.permute(0, 2, 1))
        # input = F.relu(input)  # .permute(0, 2, 1))
        # input = input.view(in_shape[0], -1).contiguous()
        # input = torch.cat((torch.sin(input), torch.cos(input)), -1)
        # output = torch.sigmoid(self.out(input))
        input = input.permute(0, 2, 1)
        for layer in self.layers:
            input = F.leaky_relu(layer(input))
        # input = input.permute(0, 2, 1)
        output = input.view(in_shape[0], -1).contiguous()
        output = output.narrow(1, 0, self.output.in_features)  # Crop to appropriate size
        output = self.output(output)
        output = output.view(in_shape[0], self.bptt, self.output_size).contiguous()
        return output, None

    def init_hidden(self, input):
        """Initialize hidden states."""
        if self.batch_first:
            dim_zero = self.num_layers * self.multiplier
            return torch.zeros(
                dim_zero,
                input.size(0), self.output_size, device=input.device)
        else:
            dim_zero = self.num_layers * self.multiplier
            return torch.zeros(
                dim_zero,
                input.size(1), self.hidden_size, device=input.device)

