import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torchkit.distributions import TanhNormal
from torchkit.networks import Mlp, EquiMlp
from utils import helpers as utl

from escnn import gspaces
from escnn import nn as enn

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
PROB_MIN = 1e-8


class MarkovPolicyBase(Mlp):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes,
        init_w=1e-3,
        image_encoder=None,
        **kwargs
    ):
        self.save_init_params(locals())
        self.action_dim = action_dim

        if image_encoder is None:
            self.input_size = obs_dim
        else:
            self.input_size = image_encoder.embed_size

        # first register MLP
        super().__init__(
            hidden_sizes,
            input_size=self.input_size,
            output_size=self.action_dim,
            init_w=init_w,
            **kwargs,
        )

        # then register image encoder
        self.image_encoder = image_encoder  # None or nn.Module

    def forward(self, obs):
        """
        :param obs: Observation, usually 2D (B, dim), but maybe 3D (T, B, dim)
        return action (*, dim)
        """
        x = self.preprocess(obs)
        return super().forward(x)

    def preprocess(self, obs):
        x = obs
        if self.image_encoder is not None:
            x = self.image_encoder(x)
        return x


class EquiMarkovPolicyBase(EquiMlp):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes,
        group_helper,
        init_w=1e-3,
        image_encoder=None,
        **kwargs
    ):
        self.action_dim = action_dim
        self.group_helper = group_helper

        if image_encoder is None:
            self.input_size = obs_dim
        else:
            self.input_size = image_encoder.embed_size

        # TODO: change into a general group action
        num_rotations = self.group_helper.num_rotations
        grp_act = self.group_helper.grp_act
        scaler = self.group_helper.scaler

        in_type = enn.FieldType(grp_act, obs_dim //
                                num_rotations // scaler*self.group_helper.reg_repr)
        hid_type = enn.FieldType(grp_act,  hidden_sizes[0]//num_rotations//scaler
                                 * self.group_helper.reg_repr)

        # 1 trivial_repr for g, 2 irrep for xy, 2 trivial_repr for z and r
        out_type = enn.FieldType(grp_act,
                                 self.group_helper.triv_repr
                                 + self.group_helper.irr_repr
                                 + self.group_helper.triv_repr
                                 + self.group_helper.triv_repr
                                 )
        # first register MLP
        super().__init__(
            len(hidden_sizes) + 1,
            in_type,
            hid_type,
            out_type,
        )

        # then register image encoder
        self.image_encoder = image_encoder  # None or nn.Module

    def forward(self, obs):
        """
        :param obs: Observation, usually 2D (B, dim), but maybe 3D (T, B, dim)
        return action (*, dim)
        """
        x = self.preprocess(obs)
        return super().forward(x)

    def preprocess(self, obs):
        x = obs
        if self.image_encoder is not None:
            x = self.image_encoder(x)
        return x


class DeterministicPolicy(MarkovPolicyBase):
    """
    Usage: TD3
    ```
    policy = DeterministicPolicy(...)
    action = policy(obs)
    ```
    NOTE: action space must be [-1,1]^d
    """

    def forward(
        self,
        obs,
    ):
        h = super().forward(obs)
        action = torch.tanh(h)  # map into [-1, 1]
        return action


class EquiDeterministicPolicy(EquiMarkovPolicyBase):
    """
    Usage: TD3-Equi
    ```
    policy = DeterministicPolicy(...)
    action = policy(obs)
    ```
    NOTE: action space must be [-1,1]^d
    """

    def forward(
        self,
        obs,
    ):
        h = super().forward(obs)
        action = torch.tanh(h)  # map into [-1, 1]
        return action


class TanhGaussianPolicy(MarkovPolicyBase):
    """
    Usage: SAC
    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.
    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    NOTE: action space must be [-1,1]^d
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes,
        std=None,
        init_w=1e-3,
        image_encoder=None,
        **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            obs_dim, action_dim, hidden_sizes, init_w, image_encoder, **kwargs
        )

        self.log_std = None
        self.std = std
        if std is None:  # learn std
            last_hidden_size = self.input_size
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            # initialized near zeros, https://arxiv.org/pdf/2005.05719v1.pdf fig 7.a
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:  # fix std
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(
        self,
        obs,
        reparameterize=True,
        deterministic=False,
        return_log_prob=False,
    ):
        """
        :param obs: Observation, usually 2D (B, dim), but maybe 3D (T, B, dim)
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = self.preprocess(obs)
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        if deterministic:
            action = torch.tanh(mean)
            assert (
                return_log_prob == False
            )  # NOTE: cannot be used for estimating entropy
        else:
            tanh_normal = TanhNormal(mean, std)  # (*, B, dim)
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=-1, keepdim=True)  # (*, B, 1)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()
        return action, mean, log_std, log_prob


class EquiTanhGaussianPolicy(EquiMarkovPolicyBase):
    """
    Usage: SAC-Equi
    ```
    policy = EquiTanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.
    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    NOTE: action space must be [-1,1]^d
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes,
        group_helper,
        std=None,
        init_w=1e-3,
        image_encoder=None,
        **kwargs
    ):
        super().__init__(
            obs_dim, action_dim, hidden_sizes, group_helper,
            init_w, image_encoder, **kwargs
        )

        self.log_std = None
        self.std = std
        if std is None:  # learn std
            grp_act = group_helper.grp_act
            # use trivial repr for all stds
            out_type = enn.FieldType(grp_act,
                                     action_dim*group_helper.triv_repr)
            self.last_fc_log_std = enn.R2Conv(self.hid_type,
                                              out_type,
                                              kernel_size=1)
        else:  # fix std
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(
        self,
        obs,
        reparameterize=True,
        deterministic=False,
        return_log_prob=False,
    ):
        """
        :param obs: Observation, usually 2D (B, dim), but maybe 3D (T, B, dim)
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = self.preprocess(obs)

        h = h.reshape((-1, self.in_type.size, 1, 1))
        h = self.in_type(h)

        for i, fc in enumerate(self.fcs[:-1]):
            h = fc(h)
            h = self.acts[i](h)

        mean = self.fcs[-1](h).tensor
        mean = mean.reshape([obs.shape[0], -1, self.action_dim])
        if obs.shape[0] == 1:
            mean = mean.reshape([1, self.action_dim])
        if self.std is None:
            log_std = self.last_fc_log_std(h).tensor
            log_std = log_std.reshape_as(mean)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        if deterministic:
            action = torch.tanh(mean)
            assert (
                return_log_prob is False
            )  # NOTE: cannot be used for estimating entropy
        else:
            tanh_normal = TanhNormal(mean, std)  # (*, B, dim)
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=-1, keepdim=True)  # (*, B, 1)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()
        return action, mean, log_std, log_prob


class CategoricalPolicy(MarkovPolicyBase):
    """Based on https://github.com/ku2482/sac-discrete.pytorch/blob/master/sacd/model.py
    Usage: SAC-discrete
    ```
    policy = CategoricalPolicy(...)
    action, _, _ = policy(obs, deterministic=True)
    action, _, _ = policy(obs, deterministic=False)
    action, prob, log_prob = policy(obs, deterministic=False, return_log_prob=True)
    ```
    NOTE: action space must be discrete
    """

    def forward(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        """
        :param obs: Observation, usually 2D (B, dim), but maybe 3D (T, B, dim)
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        return: action (*, B, A), prob (*, B, A), log_prob (*, B, A)
        """
        action_logits = super().forward(obs)  # (*, A)

        prob, log_prob = None, None
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)  # (*)
            assert (
                return_log_prob == False
            )  # NOTE: cannot be used for estimating entropy
        else:
            prob = F.softmax(action_logits, dim=-1)  # (*, A)
            distr = Categorical(prob)
            # categorical distr cannot reparameterize
            action = distr.sample()  # (*)
            if return_log_prob:
                log_prob = torch.log(torch.clamp(prob, min=PROB_MIN))

        # convert to one-hot vectors
        action = F.one_hot(action.long(), num_classes=self.action_dim).float()  # (*, A)

        return action, prob, log_prob
