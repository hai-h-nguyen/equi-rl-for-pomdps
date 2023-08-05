import gym
import torch
import torchkit.pytorch_utils as ptu

from policies.models.recurrent_critic import Critic_RNN
from torchkit.networks import EquiImageEncoder
from policies.rl.sac import SAC

from utils import helpers as utl

from escnn import gspaces
from escnn import nn as enn

import gym
import envs.pomdp

import pytest

cuda_id = 0  # -1 if using cpu
ptu.set_gpu_mode(torch.cuda.is_available() and cuda_id >= 0, cuda_id)

action_embedding_size = 8
rnn_hidden_size = 128
observ_embedding_size = 0
policy_layers = [128, 128]
rnn_num_layers = 1

test_dihedral = False

env_name = "BlockPicking-Symm-v0"
env = gym.make(env_name)
(num_channels, image_width, image_height) = env.img_size
obs_dim = num_channels*image_width*image_height
act_dim = env.action_space.shape[0]

image_encoder = {"from_flattened": False}

if test_dihedral:
    num_rotations = [4]
else:
    num_rotations = [4, 8]


def check_critics(num_rotation):

    group_helper = utl.GroupHelper(num_rotation, flip_symmetry=test_dihedral)
    image_encoder_fn = lambda: EquiImageEncoder(
                image_shape=(2, image_height, image_width),
                group_helper=group_helper,
                **image_encoder
            )

    algos = [SAC]

    for algo in algos:
        critic = Critic_RNN(
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

        utl.count_parameters(critic)

        seq_len = 51
        batch_size = 32

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

        # # seq_len = acts.shape[0]
        # # batch_size = 1

        # # acts = acts.reshape((seq_len, batch_size, act_dim)).to(ptu.device)
        # # obses = obses.reshape((seq_len, batch_size, obs_dim)).to(ptu.device)
        # # rewards = rewards.reshape((seq_len, batch_size, 1)).to(ptu.device)

        acts = torch.randn(seq_len, batch_size, act_dim).to(ptu.device)
        curr_acts = torch.randn(seq_len, batch_size, act_dim).to(ptu.device)
        obses = torch.randn(seq_len, batch_size, obs_dim).to(ptu.device)
        rewards = torch.randn(seq_len, batch_size, obs_dim).to(ptu.device)

        q1, q2 = critic(acts, rewards, obses, curr_acts)
        q1_or = q1
        q2_or = q2

        num_rotations = 4

        if test_dihedral:
            grp_act = gspaces.flipRot2dOnR2(num_rotations)
        else:
            grp_act = gspaces.rot2dOnR2(num_rotations)

        acts = acts.reshape((seq_len*batch_size, act_dim, 1, 1))
        curr_acts = curr_acts.reshape((seq_len*batch_size, act_dim, 1, 1))
        obses = obses.reshape((seq_len*batch_size, num_channels, image_width, image_height))

        if test_dihedral:
            irrep = 1*[grp_act.irrep(1, 1)]
        else:
            irrep = 1*[grp_act.irrep(1)]

        act_in_type = enn.FieldType(grp_act,  1*[grp_act.trivial_repr]
                                    + irrep
                                    + 1*[grp_act.trivial_repr]
                                    + 1*[grp_act.trivial_repr])
        acts = act_in_type(acts)
        curr_acts = act_in_type(curr_acts)

        obs_in_type = enn.FieldType(grp_act, num_channels*[grp_act.trivial_repr])
        obses = obs_in_type(obses)

        # for each group element
        with torch.no_grad():
            for g in grp_act.testing_elements:
                acts_transformed = acts.transform(g)
                curr_acts_transformed = curr_acts.transform(g)
                obses_transformed = obses.transform(g)

                acts_transformed = acts_transformed.tensor.reshape((seq_len, batch_size, act_dim))
                curr_acts_transformed = curr_acts_transformed.tensor.reshape((seq_len, batch_size, act_dim))
                obses_transformed = obses_transformed.tensor.reshape((seq_len, batch_size, obs_dim))

                q1_new, q2_new = critic(acts_transformed, rewards, obses_transformed, curr_acts_transformed)

                assert torch.allclose(q1_or, q1_new, atol=1e-5), g
                assert torch.allclose(q2_or, q2_new, atol=1e-5), g


def test_critics():

    for num_rotation in num_rotations:
        check_critics(num_rotation)

if __name__ == "__main__":
    test_critics()