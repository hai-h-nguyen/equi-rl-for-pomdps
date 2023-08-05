import torch
from torch.optim import Adam
import numpy as np
from .base import RLAlgorithmBase
from policies.models.actor import TanhGaussianPolicy, EquiTanhGaussianPolicy
from torchkit.networks import FlattenMlp, EquiFlattenMlp
import torchkit.pytorch_utils as ptu

from escnn import gspaces
from escnn import nn as enn


class SAC(RLAlgorithmBase):
    name = "sac"
    continuous_action = True
    use_target_actor = False

    def __init__(
        self,
        entropy_alpha=0.1,
        automatic_entropy_tuning=True,
        target_entropy=None,
        alpha_lr=3e-4,
        init_alpha=0.1,
        action_dim=None,
        group_helper=None,
    ):

        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.group_helper = group_helper
        if self.automatic_entropy_tuning:
            if target_entropy is not None:
                self.target_entropy = float(target_entropy)
            else:
                self.target_entropy = -float(action_dim)
            self.log_alpha_entropy = torch.tensor(
                np.log(init_alpha), requires_grad=True, device=ptu.device
            )
            self.alpha_lr = alpha_lr
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
            self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()
        else:
            self.alpha_entropy = entropy_alpha

    def get_special_dict(self):
        if self.automatic_entropy_tuning:
            dict = {}
            dict["log_alpha_entropy"] = self.log_alpha_entropy
            dict["alpha_entropy"] = self.alpha_entropy
            dict["alpha_entropy_optim"] = self.alpha_entropy_optim.state_dict()
            return dict
        else:
            return None

    def load_special_dict(self, dict):
        if self.automatic_entropy_tuning:
            self.log_alpha_entropy = dict["log_alpha_entropy"]
            self.alpha_entropy = dict["alpha_entropy"]
            # this is required
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy],
                                            lr=self.alpha_lr)
            self.alpha_entropy_optim.load_state_dict(dict["alpha_entropy_optim"])

    def update_others(self, current_log_probs):
        if self.automatic_entropy_tuning:
            alpha_entropy_loss = -self.log_alpha_entropy.exp() * (
                current_log_probs + self.target_entropy
            )

            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_entropy_optim.step()
            self.alpha_entropy = self.log_alpha_entropy.exp().item()

        return {"policy_entropy": -current_log_probs, "alpha": self.alpha_entropy}

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes,
                    group_helper, **kwargs):
        return TanhGaussianPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    @staticmethod
    def build_equi_actor(input_size, action_dim, hidden_sizes, group_helper, **kwargs):
        return EquiTanhGaussianPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            group_helper=group_helper,
            **kwargs,
        )

    @staticmethod
    def build_critic(hidden_sizes, group_helper=None,
                     input_size=None, obs_dim=None, action_dim=None):
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        qf1 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        qf2 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        return qf1, qf2

    @staticmethod
    def build_equi_critic(hidden_sizes, group_helper, input_size=None,
                          obs_dim=None, action_dim=None):
        assert action_dim is not None
        if obs_dim is not None:
            input_size = obs_dim

        num_rotations = group_helper.num_rotations
        grp_act = group_helper.grp_act

        scaler = group_helper.scaler

        in_type = enn.FieldType(grp_act,  input_size//num_rotations//scaler
                                * group_helper.reg_repr)
        hid_type = enn.FieldType(grp_act,  hidden_sizes[0]//num_rotations//scaler
                                 * group_helper.reg_repr)
        out_type = enn.FieldType(grp_act,
                                 group_helper.triv_repr)

        qf1 = EquiFlattenMlp(
            len(hidden_sizes) + 1, in_type, hid_type, out_type
        )
        qf2 = EquiFlattenMlp(
            len(hidden_sizes) + 1, in_type, hid_type, out_type
        )

        return qf1, qf2

    def select_action(self, actor, observ, deterministic: bool, return_log_prob: bool):
        return actor(observ, False, deterministic, return_log_prob)

    @staticmethod
    def forward_actor(actor, observ):
        new_actions, mean, log_stds, log_probs = actor(observ, return_log_prob=True)
        return new_actions, log_probs, mean, log_stds  # (T+1, B, dim), (T+1, B, 1)

    def critic_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions,
        rewards,
        dones,
        gamma,
        next_observs=None,  # used in markov_critic
    ):
        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from current policy,
            if markov_actor:
                new_actions, new_log_probs = self.forward_actor(
                    actor, next_observs if markov_critic else observs
                )
            else:
                # (T+1, B, dim) including reaction to last obs
                new_actions, new_log_probs, _, _ = actor(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=next_observs if markov_critic else observs,
                )

            if markov_critic:  # (B, 1)
                next_q1 = critic_target[0](next_observs, new_actions)
                next_q2 = critic_target[1](next_observs, new_actions)
            else:
                next_q1, next_q2 = critic_target(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=new_actions,
                )  # (T+1, B, 1)

            min_next_q_target = torch.min(next_q1, next_q2)
            min_next_q_target += self.alpha_entropy * (-new_log_probs)  # (T+1, B, 1)

            # q_target: (T, B, 1)
            q_target = rewards + (1.0 - dones) * gamma * min_next_q_target  # next q
            if not markov_critic:
                q_target = q_target[1:]  # (T, B, 1)

        if markov_critic:
            q1_pred = critic[0](observs, actions)
            q2_pred = critic[1](observs, actions)
        else:
            # Q(h(t), a(t)) (T, B, 1)
            q1_pred, q2_pred = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=actions[1:],
            )  # (T, B, 1)

        return (q1_pred, q2_pred), q_target

    def actor_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions=None,
        rewards=None,
        expert_masks=None,
    ):
        if markov_actor:
            new_actions, log_probs = self.forward_actor(actor, observs)
        else:
            new_actions, log_probs, _, _ = actor(
                prev_actions=actions, rewards=rewards, observs=observs
            )  # (T+1, B, A)

        if markov_critic:
            q1 = critic[0](observs, new_actions)
            q2 = critic[1](observs, new_actions)
        else:
            q1, q2 = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=new_actions,
            )  # (T+1, B, 1)
        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1)

        policy_loss = -min_q_new_actions
        policy_loss += self.alpha_entropy * log_probs
        if not markov_critic:
            policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs

        return policy_loss, log_probs

    #### Below are used in shared RNN setting
    def forward_actor_in_target(self, actor, actor_target, next_observ):
        return self.forward_actor(actor, next_observ)

    def entropy_bonus(self, log_probs):
        return self.alpha_entropy * (-log_probs)
