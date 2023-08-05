from utils.equi_conv_lstm_hzzone import ConvLSTM
import torch
from escnn import gspaces
from escnn import nn as enn

input_size = 40
hidden_size = 128
seq_len = 50
bsize = 32

num_rotations = 4
test_dihedral = False

if test_dihedral:
    grp_act = gspaces.flipRot2dOnR2(num_rotations)
    scaler = 2
else:
    grp_act = gspaces.rot2dOnR2(num_rotations)
    scaler = 1
    
in_type = enn.FieldType(grp_act, input_size//num_rotations//scaler*[grp_act.regular_repr])

rnn = ConvLSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=1,
    aux_weights=True,
    batch_first=False,
    bias=False
)

x = torch.randn((seq_len, bsize, in_type.size))

# Untransformed input
y, (h, c) = rnn(x)
y_original = y
h_original = h
c_original = c

# Transformed input
x = x.reshape((seq_len*bsize, input_size, 1, 1))
x = in_type(x)

with torch.no_grad():
    # for each group element
    for g in grp_act.testing_elements:
        x_transformed = x.transform(g)
        x_transformed = x_transformed.tensor.reshape((seq_len, bsize, in_type.size))

        y, (h, c) = rnn(x_transformed)

        assert torch.allclose(y.tensor, y_original.transform(g).tensor, atol=1e-5), g
        assert torch.allclose(h.tensor, h_original.transform(g).tensor, atol=1e-5), g
        assert torch.allclose(c.tensor, c_original.transform(g).tensor, atol=1e-5), g
