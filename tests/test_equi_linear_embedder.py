import torch
from utils import helpers as utl
from torch.nn import functional as F
import numpy as np

from escnn import gspaces
from escnn import nn as enn

num_rotations = 4

test_dihedral = True

if test_dihedral:
    grp_act = gspaces.flipRot2dOnR2(num_rotations)
    scaler = 2
else:
    grp_act = gspaces.rot2dOnR2(num_rotations)
    scaler = 1

seq_len = 50
batch_size = 32

# For actions only
action_dim = 8
embedding_size = 32

in_type = enn.FieldType(grp_act,  action_dim//num_rotations//scaler*[grp_act.regular_repr])
out_type = enn.FieldType(grp_act, embedding_size//num_rotations//scaler*[grp_act.regular_repr])

embedder = utl.EquiFeatureExtractor(in_type, out_type)

x = torch.randn(seq_len, batch_size, action_dim)
y = embedder(x)

or_x = x
or_y = y

x = x.reshape((seq_len*batch_size, action_dim, 1, 1))
x = in_type(x)

y = y.reshape((seq_len*batch_size, embedding_size, 1, 1))
y = out_type(y)

# for each group element
with torch.no_grad():
    for g in grp_act.testing_elements:
        x_transformed = x.transform(g)
        y_new = embedder(x_transformed.tensor.reshape_as(or_x))

        y_transformed = y.transform(g).tensor.reshape_as(or_y)
        assert torch.allclose(y_new, y_transformed, atol=1e-5), g
