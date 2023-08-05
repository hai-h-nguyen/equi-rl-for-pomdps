from torch import nn
import torch
import torchkit.pytorch_utils as ptu

from escnn import gspaces
from escnn import nn as enn


class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers, batch_first, bias,
                 width=1, height=1,
                 kernel_size=1, stride=1,
                 aux_weights=True):
        super().__init__()

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._width = width
        self._height = height
        self._stride = stride
        self._kernel_size = kernel_size
        self._aux_weights = aux_weights

        # Initialize weights for Hadamard Products
        if self._aux_weights:
            self.W_ci = nn.Parameter(torch.zeros(1, hidden_size, height, width))
            self.W_co = nn.Parameter(torch.zeros(1, hidden_size, height, width))
            self.W_cf = nn.Parameter(torch.zeros(1, hidden_size, height, width))

    def init_group(self, group_helper):
        num_rotations = group_helper.num_rotations
        grp_act = group_helper.grp_act
        scaler = group_helper.scaler
        self.in_type = enn.FieldType(grp_act,
                                     (self._input_size + self._hidden_size)
                                     // num_rotations // scaler
                                     * group_helper.reg_repr)

        divisor = num_rotations // 4

        self.out_type = enn.FieldType(grp_act, self._hidden_size // divisor // scaler
                                      * group_helper.reg_repr)

        self.c_type = enn.FieldType(grp_act, self._hidden_size
                                    // num_rotations // scaler
                                    * group_helper.reg_repr)

        self._conv = enn.R2Conv(self.in_type,
                                self.out_type,
                                stride=self._stride,
                                padding=self._kernel_size//2,
                                kernel_size=1)

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

            if isinstance(h, enn.GeometricTensor):
                h = h.tensor
                c = c.tensor

        outputs = []
        for t in range(seq_len):
            x = inputs[t, ...]
            cat_x = torch.cat([x, h], dim=1)
            cat_x = self.in_type(cat_x)
            conv_x = self._conv(cat_x)

            # chunk across channel dimension
            i, f, g, o = torch.chunk(conv_x.tensor, chunks=4, dim=1)

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

        outputs = torch.stack(outputs)

        outputs = outputs.reshape(seq_len*batch_size, -1, height, width)

        return self.c_type(outputs), (self.c_type(h), self.c_type(c))
