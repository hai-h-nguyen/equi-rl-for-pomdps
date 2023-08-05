import torch
from torch.nn import functional as F

from escnn import gspaces
from escnn import nn as enn
from torchkit.networks import EquiImageEncoder

import gym
import envs.pomdp

seq_len = 51
batch_size = 32
num_rotations = 4
grp_act = gspaces.rot2dOnR2(num_rotations)

env_name = "CarFlag-Symm-2d-v0"
env = gym.make(env_name)
(num_channels, image_width, image_height) = env.img_size
obs_dim = num_channels*image_width*image_height
act_dim = env.action_space.n

embedding_size = 128

image_shape = (num_channels, image_height, image_width)
embedder = EquiImageEncoder(image_shape, from_flattened=True)

in_type = enn.FieldType(grp_act,  num_channels*[grp_act.trivial_repr])
out_type = enn.FieldType(grp_act, embedding_size//num_rotations*[grp_act.regular_repr])

x = torch.randn(seq_len, batch_size, num_channels*image_height*image_width)
y = embedder(x)

or_x = x

x = x.reshape((seq_len*batch_size, num_channels, image_height, image_width))
x = in_type(x)

y = y.reshape((seq_len*batch_size, embedding_size, 1, 1))
y = out_type(y)

# for each group element
with torch.no_grad():
    for g in grp_act.testing_elements:
        x_new = x.transform(g)
        y_new = embedder(x_new.tensor.reshape_as(or_x))

        y_transformed = y.transform(g).tensor.reshape_as(y_new)

        assert torch.allclose(y_new, y_transformed, atol=1e-5), g
