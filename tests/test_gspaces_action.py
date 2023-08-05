from escnn import gspaces
from escnn import nn
import torch              
import numpy as np

grp_act = gspaces.rot2dOnR2(4)
in_type  = nn.FieldType(grp_act, [grp_act.regular_repr])

action = [1, 0, 0, 0]
x = torch.tensor(action).view(1, 4, 1, 1)
x = in_type(x)

print(x.tensor.squeeze())

for g in grp_act.testing_elements:
    print(g, x.transform(g).tensor.squeeze())
