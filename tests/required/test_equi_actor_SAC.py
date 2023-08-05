import gym
import numpy as np
import torch
import torchkit.pytorch_utils as ptu

from policies.models.recurrent_actor import Actor_RNN
from policies.rl.sac import SAC
from torchkit.networks import EquiImageEncoder
from utils import helpers as utl

from escnn import gspaces
from escnn import nn as enn

import envs.pomdp

import pytest


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


cuda_id = 0  # -1 if using cpu
ptu.set_gpu_mode(torch.cuda.is_available() and cuda_id >= 0, cuda_id)

action_embedding_size = 8
rnn_hidden_size = 128
observ_embedding_size = 0  # use a separate image encoder
policy_layers = [128, 128]
rnn_num_layers = 1

test_dihedral = True

env_name = "BlockPicking-Symm-v0"
env = gym.make(env_name)
(num_channels, image_width, image_height) = env.img_size
obs_dim = num_channels*image_width*image_height
act_dim = env.action_space.shape[0]

if test_dihedral:
    num_rotations = [4]
else:
    num_rotations = [4, 8]


def check_actor(num_rotation):

    group_helper = utl.GroupHelper(num_rotation, flip_symmetry=test_dihedral)

    image_encoder = {"from_flattened": False}
    image_encoder_fn = lambda: EquiImageEncoder(
                    image_shape=env.img_size,
                    group_helper=group_helper,
                    **image_encoder
                )

    algo = SAC

    equi_actor = Actor_RNN(
        obs_dim,
        act_dim,
        "equi",
        algo,
        action_embedding_size,
        observ_embedding_size,
        rnn_hidden_size,
        policy_layers,
        rnn_num_layers,
        image_encoder=image_encoder_fn(),  # separate weight
        group_helper=group_helper,
    ).to(ptu.device)

    normal_actor = Actor_RNN(
        obs_dim,
        act_dim,
        "normal",
        algo,
        action_embedding_size,
        observ_embedding_size,
        rnn_hidden_size,
        policy_layers,
        rnn_num_layers,
        image_encoder=None,  # separate weight
    ).to(ptu.device)

    utl.count_parameters(equi_actor)
    # utl.count_parameters(normal_actor)

    seq_len = 51
    batch_size = 32
    num_rotations = 4

    # Generate some data

    max_trajectory_len = env._max_episode_steps

    done = False
    acts_lst = []
    obses_lst = []
    rewards_lst = []

    env.reset()

    while not done:
        act = env.action_space.sample()
        obs, reward, done, _ = env.step(act)

        acts_lst.append(act)
        obses_lst.append(obs)
        rewards_lst.append(reward)

    acts = torch.FloatTensor(acts_lst).to(ptu.device)
    obses = torch.FloatTensor(obses_lst).to(ptu.device)
    rewards = torch.FloatTensor(rewards_lst).to(ptu.device)

    # seq_len = acts.shape[0]
    # batch_size = 1

    # acts = acts.reshape((seq_len, batch_size, act_dim)).to(ptu.device)
    # obses = obses.reshape((seq_len, batch_size, obs_dim)).to(ptu.device)
    # rewards = rewards.reshape((seq_len, batch_size, 1)).to(ptu.device)
    acts = torch.randn(seq_len, batch_size, act_dim).to(ptu.device)
    obses = torch.randn(seq_len, batch_size, obs_dim).to(ptu.device)
    rewards = torch.randn(seq_len, batch_size, 1).to(ptu.device)

    _, _, mean, log_stds = equi_actor(acts, rewards, obses)  # 51, 32, 4

    grp_act = gspaces.rot2dOnR2(num_rotations)
    mean_out_type = enn.FieldType(grp_act,  1*[grp_act.trivial_repr]
                                  + 1*[grp_act.irrep(1)]
                                  + 1*[grp_act.trivial_repr]
                                  + 1*[grp_act.trivial_repr])
    log_stds_out_type = enn.FieldType(grp_act,  act_dim*[grp_act.trivial_repr])

    mean = mean.reshape([seq_len*batch_size, act_dim, 1, 1])
    log_stds = log_stds.reshape([seq_len*batch_size, act_dim, 1, 1])

    mean_or = mean_out_type(mean)
    log_stds_or = log_stds_out_type(log_stds)

    acts = acts.reshape((seq_len*batch_size, act_dim, 1, 1)).to(ptu.device)
    obses = obses.reshape((seq_len*batch_size, num_channels, image_width, image_height)).to(ptu.device)

    act_in_type = enn.FieldType(grp_act,  1*[grp_act.trivial_repr]
                                + 1*[grp_act.irrep(1)]
                                + 1*[grp_act.trivial_repr]
                                + 1*[grp_act.trivial_repr])
    acts = act_in_type(acts)

    obs_in_type = enn.FieldType(grp_act, num_channels*[grp_act.trivial_repr])
    obses = obs_in_type(obses)

    # for each group element
    with torch.no_grad():
        for g in grp_act.testing_elements:
            acts_transformed = acts.transform(g)
            obses_transformed = obses.transform(g)

            acts_transformed = acts_transformed.tensor.reshape((seq_len, batch_size, act_dim))
            obses_transformed = obses_transformed.tensor.reshape((seq_len, batch_size, obs_dim))

            _, _, mean_new, log_stds_new = equi_actor(acts_transformed, rewards, obses_transformed)

            mean = mean_or.transform(g).tensor.reshape_as(mean_new)
            log_stds = log_stds_or.transform(g).tensor.reshape_as(log_stds_new)

            assert torch.allclose(mean_new, mean, atol=1e-5), g
            assert torch.allclose(log_stds_new, log_stds, atol=1e-5), g


def test_actor():
    for num_rotation in num_rotations:
        check_actor(num_rotation)


if __name__ == "__main__":
    test_actor()