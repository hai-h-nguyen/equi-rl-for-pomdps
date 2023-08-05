from torchkit.networks import EquiFlattenMlp
import torch
import numpy as np
from utils import helpers as utl

from escnn import nn as enn
from escnn import gspaces

test_dihedral = True

input_dim = 256
hidden_dims = [128, 128]
output_dim = 8
num_rotations = 4
seq_len = 50
batch_size = 32

if test_dihedral:
    grp_act = gspaces.flipRot2dOnR2(num_rotations)
    scaler = 2
else:
    grp_act = gspaces.rot2dOnR2(num_rotations)
    scaler = 1
    
in_type = enn.FieldType(grp_act,  input_dim//num_rotations//scaler*[grp_act.regular_repr])
hid_type = enn.FieldType(grp_act,  hidden_dims[0]//num_rotations//scaler*[grp_act.regular_repr])
out_type = enn.FieldType(grp_act,  output_dim//num_rotations//scaler*[grp_act.regular_repr])

qf1 = EquiFlattenMlp(
    len(hidden_dims) + 1, in_type, hid_type, out_type
)

utl.count_parameters(qf1)

input = torch.rand(seq_len, batch_size, input_dim)

q1 = qf1(input)

q1 = q1.reshape(seq_len*batch_size, output_dim, 1, 1)

q1_or = out_type(q1)

# for each group element
with torch.no_grad():
    for g in grp_act.testing_elements:
        input = input.reshape(seq_len*batch_size, input_dim, 1, 1)
        input_new = in_type(input)
        input_new = input_new.transform(g)
        input_new = input_new.tensor.reshape(seq_len, batch_size, input_dim)

        q1_new = qf1(input_new)

        q1 = q1_or.transform(g)

        q1 = q1.tensor.reshape_as(q1_new)

        assert torch.allclose(q1_new, q1, atol=1e-5), g
