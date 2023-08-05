from torch import nn
import torch
import torchkit.pytorch_utils as ptu
import time


class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers, batch_first, bias,
                 width=1, height=1,
                 kernel_size=1, stride=1,
                 aux_weights=False):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=input_size + hidden_size,
                               out_channels=hidden_size*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size//2)

        if not aux_weights:
            torch.nn.init.xavier_uniform(self._conv.weight)

        self._hidden_size = hidden_size
        self._width = width
        self._height = height

        self._aux_weights = aux_weights

        # Initialize weights for Hadamard Products
        if self._aux_weights:
            self.W_ci = nn.Parameter(torch.zeros(1, hidden_size, height, width))
            self.W_co = nn.Parameter(torch.zeros(1, hidden_size, height, width))
            self.W_cf = nn.Parameter(torch.zeros(1, hidden_size, height, width))

    def forward(self, inputs, states=None):
        seq_len = inputs.size(0)
        batch_size = inputs.size(1)
        num_channel = inputs.size(2)
        height = self._height
        width = self._width

        # seq_len (S), batch_size (B), num_channel (C), height (H), width (W)
        inputs = inputs.reshape(seq_len, batch_size,
                                num_channel, height, width)

        if states is None:
            c = torch.zeros((batch_size, self._hidden_size, height,
                             width), dtype=torch.float).to(ptu.device)
            h = torch.zeros((batch_size, self._hidden_size, height,
                             width), dtype=torch.float).to(ptu.device)
        else:
            h, c = states

        outputs = []
        for t in range(seq_len):
            x = inputs[t, ...]
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)

            # chunk across channel dimension
            i, f, g, o = conv_x.chunk(4, dim=1)

            if self._aux_weights:
                i = torch.sigmoid(i + self.W_ci*c)
                f = torch.sigmoid(f + self.W_cf*c)
            else:
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)

            c = f*c + i*torch.tanh(g)

            if self._aux_weights:
                o = torch.sigmoid(o + self.W_co*c)
            else:
                o = torch.sigmoid(o)
            h = o*torch.tanh(c)

            outputs.append(h)

        return torch.stack(outputs), (h, c)
