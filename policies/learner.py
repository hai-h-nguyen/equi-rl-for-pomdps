# -*- coding: future_fstrings -*-
import os
import time

import math
import numpy as np
import random
import torch
from torch.nn import functional as F
import gym

from utils.helpers import center_crop

# suppress this warning https://github.com/openai/gym/issues/1844
gym.logger.set_level(40)

import joblib

from .models import AGENT_CLASSES, AGENT_ARCHS
from torchkit.networks import ImageEncoder, EquiImageEncoder

# Markov policy
from buffers.simple import SimpleBuffer

from buffers.seq_vanilla import SeqBuffer
from buffers.seq_rot import SeqRotBuffer
from buffers.seq_per_rot import SeqPerRotBuffer
from buffers.seq_rad_rot import SeqRadRotBuffer

from buffers_efficient.seq_vanilla import SeqBuffer as SeqBufferEff
from buffers_efficient.seq_rot import SeqRotBuffer as SeqRotBufferEff
from buffers_efficient.seq_per_rot import SeqPerRotBuffer as SeqPerRotBufferEff

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from utils import logger

import wandb


class Learner:
    def __init__(self, env_args, train_args, eval_args,
                 policy_args, seed, replay, time_limit,
                 prefix, ckpt_dir, cfg_file, **kwargs):
        self.seed = seed
        self.group_prefix = prefix

        self.replay = replay

        self.time_limit = time_limit

        # TODO:
        self.per_expert_eps = 1.0
        self.per_eps = 1e-6

        ckpt_filename = f"{env_args['env_name'][:-3]}"      \
                        + f"_{policy_args['algo_name']}"    \
                        + f"_{policy_args['actor_type']}"   \
                        + f"_{policy_args['critic_type']}"  \
                        + f"_r{policy_args['num_rotations']}"\
                        + f"_e{train_args['num_expert_rollouts_pool']}"\
                        + f"_s{self.seed}.pt"

        # For RAD environment, observation must be center-cropped
        self.crop_obs = 'RAD' in env_args['env_name']

        if ckpt_dir is not None:
            self.checkpoint_dir = os.path.join(ckpt_dir, ckpt_filename)
        else:
            self.checkpoint_dir = ckpt_filename

        self.checkpoint_dir_end = self.checkpoint_dir + '_end'

        if os.path.exists(self.checkpoint_dir):
            self.chkpt_dict = joblib.load(self.checkpoint_dir)
            logger.log("Load checkpoint file done")
            self.set_random_state()
        else:
            # this file exists, then this run has ended
            if os.path.exists(self.checkpoint_dir_end):
                logger.log("End file found, exit")
                exit(0)
            logger.log("Checkpoint file not found")
            self.chkpt_dict = None

        if not replay:
            self.init_wandb(cfg_file, **env_args, **policy_args, **train_args)

        self.init_env(**env_args)

        self.init_agent(**policy_args)

        self.init_train(**train_args)

        self.init_eval(**eval_args)

    def set_random_state(self):
        random.setstate(self.chkpt_dict["random_rng_state"])
        np.random.set_state(self.chkpt_dict["numpy_rng_state"])
        torch.set_rng_state(self.chkpt_dict["torch_rng_state"])
        torch.cuda.set_rng_state(self.chkpt_dict["torch_cuda_rng_state"])
        logger.log("Load random state checkpoint done")

    def init_env(
        self,
        env_type,
        env_name,
        max_rollouts_per_task=None,
        num_tasks=None,
        num_train_tasks=None,
        num_eval_tasks=None,
        eval_envs=None,
        worst_percentile=None,
        **kwargs
    ):

        # initialize environment
        assert env_type in [
            "pomdp",
        ]
        self.env_type = env_type

        if self.env_type in [
            "pomdp",
        ]:  # pomdp/mdp task, using pomdp wrapper
            import envs.pomdp

            assert num_eval_tasks > 0
            self.train_env = gym.make(env_name, rendering=self.replay)
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)  # crucial

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = 50

        else:
            raise ValueError

        # get action / observation dimensions
        if self.train_env.action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = self.train_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.train_env.action_space.__class__.__name__ == "Discrete"
            self.act_dim = self.train_env.action_space.n
            self.act_continuous = False
        if len(self.train_env.observation_space.shape) == 3:
            self.obs_dim = self.train_env.observation_space.shape
        else:
            self.obs_dim = self.train_env.observation_space.shape[0]  # include 1-dim done

        logger.log("obs_dim", self.obs_dim, "act_dim", self.act_dim)

    def init_agent(
        self,
        actor_type,
        critic_type,
        image_encoder=None,
        num_rotations=4,
        flip_symmetry=False,
        **kwargs
    ):
        # initialize agent

        agent_class = AGENT_CLASSES["Policy_Separate_RNN"]

        self.agent_arch = agent_class.ARCH

        if self.chkpt_dict is None:
            logger.log(agent_class, self.agent_arch)

        if actor_type == 'equi' or critic_type == 'equi':
            group_helper = utl.GroupHelper(num_rotations=num_rotations, flip_symmetry=flip_symmetry)
        else:
            group_helper = None

        if image_encoder is not None:  # carflag-symm-2d
            if actor_type == 'equi':
                actor_encoder = EquiImageEncoder
            else:
                actor_encoder = ImageEncoder

            actor_image_encoder_fn = lambda: actor_encoder(
                image_shape=self.train_env.image_space.shape, **image_encoder,
                group_helper=group_helper
            )

            if critic_type == 'equi':
                critic_encoder = EquiImageEncoder
            else:
                critic_encoder = ImageEncoder

            critic_image_encoder_fn = lambda: critic_encoder(
                image_shape=self.train_env.image_space.shape, **image_encoder,
                group_helper=group_helper
            )

        else:
            actor_image_encoder_fn = lambda: None
            critic_image_encoder_fn = lambda: None

        self.agent = agent_class(
            actor_type=actor_type,
            critic_type=critic_type,
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            actor_image_encoder_fn=actor_image_encoder_fn,
            critic_image_encoder_fn=critic_image_encoder_fn,
            group_helper=group_helper,
            **kwargs,
        ).to(ptu.device)

        if self.chkpt_dict is not None:
            self.agent.restore_state_dict(self.chkpt_dict["agent_dict"])
            logger.log("Load agent checkpoint done")

        if self.chkpt_dict is None:
            logger.log(self.agent)

    def init_train(
        self,
        buffer_size,
        batch_size,
        num_iters,
        num_init_rollouts_pool,
        num_expert_rollouts_pool,
        num_rollouts_per_iter,
        num_updates_per_iter=None,
        sampled_seq_len=None,
        sample_weight_baseline=None,
        num_aug_episode=None,
        buffer_type=None,
        **kwargs
    ):

        if num_updates_per_iter is None:
            num_updates_per_iter = 1.0
        assert isinstance(num_updates_per_iter, int) or isinstance(
            num_updates_per_iter, float
        )
        # if int, it means absolute value; if float, it means the multiplier of collected env steps
        self.num_updates_per_iter = num_updates_per_iter

        if self.agent_arch == AGENT_ARCHS.Markov:
            self.policy_storage = SimpleBuffer(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                max_trajectory_len=self.max_trajectory_len,
                add_timeout=False,  # no timeout storage
            )

        else:  # memory, memory-markov
            if sampled_seq_len == -1:
                sampled_seq_len = self.max_trajectory_len

            # no rotational augmentation
            if buffer_type == SeqBuffer.buffer_type:
                buffer_class = SeqBuffer

            # rotational augmentation
            elif buffer_type == SeqRotBuffer.buffer_type:
                buffer_class = SeqRotBuffer

            # prioritized replay + rotational augmentation (for BlockPushing)
            elif buffer_type == SeqPerRotBuffer.buffer_type:
                buffer_class = SeqPerRotBuffer

            # RAD
            elif buffer_type == SeqRadRotBuffer.buffer_type:
                buffer_class = SeqRadRotBuffer

            # no rotational augmentation efficient
            elif buffer_type == SeqBufferEff.buffer_type:
                buffer_class = SeqBufferEff

            # rotational augmentation efficient
            elif buffer_type == SeqRotBufferEff.buffer_type:
                buffer_class = SeqRotBufferEff

            # prioritized replay + rotational augmentation (for BlockPushing)
            elif buffer_type == SeqPerRotBufferEff.buffer_type:
                buffer_class = SeqPerRotBufferEff

            else:
                raise NotImplementedError

            if self.chkpt_dict is None:
                logger.log(buffer_class)

            self.policy_storage = buffer_class(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                sampled_seq_len=sampled_seq_len,
                sample_weight_baseline=sample_weight_baseline,
                num_aug_episode=num_aug_episode,
                observation_type=self.train_env.observation_space.dtype,
            )

        # load buffer from checkpoint
        if self.chkpt_dict is not None:
            self.policy_storage.load_from_state_dict(
                    self.chkpt_dict["buffer_dict"])
            logger.log("Load buffer checkpoint done")

        self.batch_size = batch_size
        self.num_iters = num_iters

        self.num_init_rollouts_pool = num_init_rollouts_pool
        self.num_expert_rollouts_pool = num_expert_rollouts_pool
        self.num_rollouts_per_iter = num_rollouts_per_iter

        total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
        self.n_env_steps_total = self.max_trajectory_len * total_rollouts
        logger.log(
            "*** total rollouts",
            total_rollouts,
            "total env steps",
            self.n_env_steps_total,
        )

    def init_eval(
        self,
        log_interval,
        save_interval,
        log_tensorboard,
        num_episodes_per_task=1,
        **kwargs
    ):

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.log_tensorboard = log_tensorboard
        self.eval_num_episodes_per_task = num_episodes_per_task

    def init_wandb(
        self,
        cfg_file,
        env_name,
        algo_name,
        actor_type,
        critic_type,
        num_rotations,
        num_expert_rollouts_pool,
        **kwargs,
    ):
        project_name = f"Symmetry_{env_name[:-3]}"

        group = f"{algo_name}_{actor_type}_{critic_type}_" + \
                f"r{num_rotations}_e{num_expert_rollouts_pool}"

        if self.group_prefix is not None:
            group = f"{self.group_prefix}_{group}"

        wandb_args = {}
        if self.chkpt_dict is not None:
            wandb_args = {"resume": "allow",
                          "id": self.chkpt_dict["wandb_id"]}
        else:
            wandb_args = {"resume": None}

        wandb.init(project=project_name,
                   settings=wandb.Settings(_disable_stats=True),
                   group=group,
                   name=f"s{self.seed}",
                   **wandb_args)
        wandb.save(cfg_file)

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._last_time = 0

        self._best_per = 0.0

        if self.chkpt_dict is not None:
            self._n_env_steps_total = self.chkpt_dict["_n_env_steps_total"]
            self._n_rollouts_total = self.chkpt_dict["_n_rollouts_total"]
            self._n_rl_update_steps_total = self.chkpt_dict["_n_rl_update_steps_total"]
            self._n_env_steps_total_last = self.chkpt_dict["_n_env_steps_total_last"]
            self._last_time = self.chkpt_dict["_last_time"]
            self._best_per = self.chkpt_dict["_best_per"]
            logger.log("Load training statistic done")
            # final load so save some memory
            self.chkpt_dict = {}

        self._start_time = time.time()
        self._start_time_last = time.time()

    def train(self, initial_model):
        """
        training loop
        """

        self._start_training()

        if initial_model is not None:
            self.load_model(initial_model)
            self.agent.reset_optimizers()

        if self.num_expert_rollouts_pool > 0 and self.chkpt_dict is None:
            logger.log("Collecting expert pool of data...")
            self.collect_expert_rollouts(
                num_rollouts=self.num_expert_rollouts_pool,
            )
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            if isinstance(self.num_updates_per_iter, float):
                # update: pomdp task updates more for the first iter_
                train_stats = self.update(
                    int(self._n_env_steps_total * self.num_updates_per_iter)
                )
                self.log_train_stats(train_stats)

        if self.num_init_rollouts_pool > 0 and self.chkpt_dict is None:
            logger.log("Collecting initial pool of data...")
            env_steps = self.collect_rollouts(
                num_rollouts=self.num_init_rollouts_pool,
                random_actions=True,
            )
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            if isinstance(self.num_updates_per_iter, float):
                # update: pomdp task updates more for the first iter_
                train_stats = self.update(
                    int(env_steps * self.num_updates_per_iter)
                )
                self.log_train_stats(train_stats)

        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            # collect data from num_rollouts_per_iter train tasks:
            env_steps = self.collect_rollouts(num_rollouts=self.num_rollouts_per_iter)
            logger.log("env steps", self._n_env_steps_total)

            # Save checkpoint
            if (time.time() - self._start_time)/3600.0 > self.time_limit:
                logger.log(f"Saving checkpoint {self.checkpoint_dir}...")

                # save replay buffer data
                buffer_dict = self.policy_storage.get_state_dict()

                self._last_time += (time.time() - self._start_time) / 3600.0

                joblib.dump(
                    {
                        "buffer_dict": buffer_dict,

                        "agent_dict": self.agent.get_state_dict(),

                        "random_rng_state": random.getstate(),
                        "numpy_rng_state": np.random.get_state(),
                        "torch_rng_state": torch.get_rng_state(),
                        "torch_cuda_rng_state": torch.cuda.get_rng_state(),

                        "_n_env_steps_total": self._n_env_steps_total,
                        "_n_rollouts_total": self._n_rollouts_total,
                        "_n_rl_update_steps_total": self._n_rl_update_steps_total,
                        "_n_env_steps_total_last": self._n_env_steps_total_last,

                        "wandb_id": wandb.run.id,

                        "_last_time": self._last_time,
                        "_best_per": self._best_per
                    },
                    self.checkpoint_dir,
                )
                logger.log("Checkpointing done and exit")
                exit(0)

            train_stats = self.update(
                self.num_updates_per_iter
                if isinstance(self.num_updates_per_iter, int)
                else int(math.ceil(self.num_updates_per_iter * env_steps))
            )  # NOTE: ceil to make sure at least 1 step
            self.log_train_stats(train_stats)

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.num_rollouts_per_iter * self.max_trajectory_len
            )
            if (
                current_num_iters != last_eval_num_iters
                and current_num_iters % self.log_interval == 0
            ):
                last_eval_num_iters = current_num_iters
                perf = self.log()

                # save best model
                if (
                    self.save_interval > 0 and
                    self._n_env_steps_total >= 0.75 * self.n_env_steps_total
                ):
                    if perf > self._best_per:
                        logger.log(f"Replace {self._best_per} w/ {perf} model")
                        self._best_per = perf
                        self.save_model(current_num_iters, perf,
                                        wandb_save=True)

                # save model according to a frequency
                if (
                    self.save_interval > 0 and
                    self._n_env_steps_total >= 0.75 * self.n_env_steps_total
                    and current_num_iters % self.save_interval == 0
                ):
                    # save models in later training stage
                    self.save_model(current_num_iters, perf, wandb_save=True)
        self.save_model(current_num_iters, perf, wandb_save=True)

        if os.path.exists(self.checkpoint_dir):
            # remove checkpoint file to save space
            os.system(f"rm {self.checkpoint_dir}")
            logger.log("Remove checkpoint file")

            # create a file to signify that this run has ended
            joblib.dump(
                    {
                        "_n_env_steps_total": self._n_env_steps_total
                    },
                    self.checkpoint_dir_end
            )

    def replay_policy(self, policy_dir):
        """
        replay a policy
        """
        self._start_training()

        self.load_model(policy_dir)

        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            # collect data from num_rollouts_per_iter train tasks:
            env_steps = self.collect_rollouts(num_rollouts=self.num_rollouts_per_iter)
            logger.log("env steps", self._n_env_steps_total)

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.num_rollouts_per_iter * self.max_trajectory_len
            )
            if (
                current_num_iters != last_eval_num_iters
                and current_num_iters % self.log_interval == 0
            ):
                last_eval_num_iters = current_num_iters

    @torch.no_grad()
    def collect_expert_rollouts(self, num_rollouts):
        """collect num_rollouts of trajectories in task using expert
        """

        before_env_steps = self._n_env_steps_total
        expert_ep_cnt = 0
        while (expert_ep_cnt < num_rollouts):
            steps = 0

            obs = ptu.from_numpy(self.train_env.reset())  # reset

            obs = obs.reshape(1, *obs.shape)
            done_rollout = False

            if self.agent_arch in [AGENT_ARCHS.Memory]:
                # temporary storage
                obs_list, act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )

            if self.agent_arch == AGENT_ARCHS.Memory:
                # get hidden state at timestep=0, None for markov
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, reward, internal_state = self.agent.get_initial_info()

            while not done_rollout:
                action = ptu.FloatTensor(
                    [self.train_env.query_expert(expert_ep_cnt)]
                )  # (1, A) for continuous action, (1) for discrete action
                if not self.act_continuous:
                    action = F.one_hot(
                        action.long(), num_classes=self.act_dim
                    ).float()  # (1, A)

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                # update statistics
                steps += 1

                # term ignore time-out scenarios, but record early stopping
                term = (
                    False
                    if "TimeLimit.truncated" in info
                    or steps >= self.max_trajectory_len
                    else done_rollout
                )

                # append tensors to temporary storage
                obs_list.append(obs)  # (1, dim)
                act_list.append(action)  # (1, dim)
                rew_list.append(reward)  # (1, dim)
                term_list.append(term)  # bool
                next_obs_list.append(next_obs)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()

            if self.agent_arch in [AGENT_ARCHS.Memory]:
                # add collected sequence to buffer
                act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)

                # only add good and short episodes
                _reward = torch.cat(rew_list, dim=0).sum().item()
                if info["success"]:
                    self.policy_storage.add_episode(
                        observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                        actions=ptu.get_numpy(act_buffer),  # (L, dim)
                        rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                        terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                        next_observations=ptu.get_numpy(
                            torch.cat(next_obs_list, dim=0)
                        ),  # (L, dim)
                        expert_masks=np.ones_like(term_list).reshape(-1, 1),  # (L, 1)
                    )

                    print(
                        f"expert steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                    )
                    self._n_env_steps_total += steps
                    self._n_rollouts_total += 1

                    expert_ep_cnt += 1
        return self._n_env_steps_total - before_env_steps

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """

        before_env_steps = self._n_env_steps_total
        for idx in range(num_rollouts):
            steps = 0

            obs = ptu.from_numpy(self.train_env.reset())  # reset

            obs = obs.reshape(1, *obs.shape)

            done_rollout = False

            if self.agent_arch in [AGENT_ARCHS.Memory]:
                # temporary storage
                obs_list, act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )

            if self.agent_arch == AGENT_ARCHS.Memory:
                # get hidden state at timestep=0, None for markov
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, reward, internal_state = self.agent.get_initial_info()

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor(
                        [self.train_env.action_space.sample()]
                    )  # (1, A) for continuous action, (1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.long(), num_classes=self.act_dim
                        ).float()  # (1, A)
                else:
                    # policy takes hidden state as input for memory-based actor,
                    # while takes obs for markov actor
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=center_crop(obs) if self.crop_obs else obs,
                            deterministic=False,
                        )
                    else:
                        action, _, _, _ = self.agent.act(obs, deterministic=False)

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )
                if self.replay:
                    time.sleep(0.1)

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                # update statistics
                steps += 1

                ## determine terminal flag per environment
                # term ignore time-out scenarios, but record early stopping
                term = (
                    False
                    if "TimeLimit.truncated" in info
                    or steps >= self.max_trajectory_len
                    else done_rollout
                )

                # add data to policy buffer
                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(
                                action.squeeze(dim=0), dim=-1, keepdims=True
                            )  # (1,)
                        ),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                    )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()

            success = False
            if self.agent_arch in [AGENT_ARCHS.Memory]:
                # add collected sequence to buffer
                act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)
                success = self.policy_storage.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                    actions=ptu.get_numpy(act_buffer),  # (L, dim)
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                    terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                    next_observations=ptu.get_numpy(
                        torch.cat(next_obs_list, dim=0)
                    ),  # (L, dim)
                    expert_masks=np.zeros_like(term_list).reshape(-1, 1),  # (L, 1)
                )

                if success:
                    print(
                        f"steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                    )

            if success:
                self._n_env_steps_total += steps
                self._n_rollouts_total += 1
        return self._n_env_steps_total - before_env_steps

    def sample_rl_batch(self, batch_size):
        """sample batch of episodes for vae training"""
        if self.agent_arch == AGENT_ARCHS.Markov:
            batch = self.policy_storage.random_batch(batch_size)
        else:  # rnn: all items are (sampled_seq_len, B, dim)
            batch, episode_indices, weights = self.policy_storage.random_episodes(batch_size)
        return ptu.np_to_pytorch_batch(batch), episode_indices, weights

    def update(self, num_updates):
        rl_losses_agg = {}
        for update in range(num_updates):
            # sample random RL batch: in transitions
            batch, episode_indices, weights = self.sample_rl_batch(self.batch_size)

            # RL update
            rl_losses = self.agent.update(batch, weights)

            # update priorities if using prioritized replay buffer
            if isinstance(self.policy_storage, SeqPerRotBuffer) or isinstance(self.policy_storage, SeqPerRotBufferEff):
                new_priorities = (rl_losses['avg_abs_td_errors'].cpu().numpy()
                                  +
                                  batch["exp_msk"].mean(dim=0).cpu().numpy() *  # bonus for expert episodes
                                  self.per_expert_eps
                                  +
                                  self.per_eps)
                self.policy_storage.update_priorities(episode_indices,
                                                      new_priorities)

            for k, v in rl_losses.items():
                if k != "avg_abs_td_errors":
                    if update == 0:  # first iterate - create list
                        rl_losses_agg[k] = [v]
                    else:  # append values
                        rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg

    @torch.no_grad()
    def evaluate(self, tasks, deterministic=True):

        num_episodes = self.max_rollouts_per_task  # k
        # max_trajectory_len = k*H
        returns_per_episode = np.zeros((len(tasks), num_episodes))
        success_rate = np.zeros(len(tasks))
        total_steps = np.zeros(len(tasks))

        num_steps_per_episode = self.eval_env._max_episode_steps
        observations = None

        for task_idx, task in enumerate(tasks):
            step = 0

            obs = ptu.from_numpy(self.eval_env.reset())  # reset

            obs = obs.reshape(1, *obs.shape)

            if self.crop_obs:
                obs = center_crop(obs)

            if self.agent_arch == AGENT_ARCHS.Memory:
                # assume initial reward = 0.0
                action, reward, internal_state = self.agent.get_initial_info()

            for episode_idx in range(num_episodes):
                running_reward = 0.0
                for _ in range(num_steps_per_episode):
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=deterministic,
                        )
                    else:
                        action, _, _, _ = self.agent.act(
                            obs, deterministic=deterministic
                        )

                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(
                        self.eval_env, action.squeeze(dim=0)
                    )

                    # add raw reward
                    running_reward += reward.item()

                    step += 1
                    done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                    if self.crop_obs:
                        next_obs = center_crop(next_obs)

                    # set: obs <- next_obs
                    obs = next_obs.clone()

                    if "success" in info and info["success"]:  # keytodoor
                        success_rate[task_idx] = 1.0

                    if done_rollout:
                        # for all env types, same
                        break

                returns_per_episode[task_idx, episode_idx] = running_reward
            total_steps[task_idx] = step
        return returns_per_episode, success_rate, observations, total_steps

    def log_train_stats(self, train_stats):
        ## log losses
        for k, v in train_stats.items():
            wandb.log({"rl_loss/" + k: v}, step=self._n_env_steps_total)
        ## gradient norms
        if self.agent_arch in [AGENT_ARCHS.Memory]:
            results = self.agent.report_grad_norm()
            for k, v in results.items():
                wandb.log({"rl_loss/" + k: v}, step=self._n_env_steps_total)

    def log(self):
        # --- evaluation ----
        returns_eval, success_rate_eval, _, total_steps_eval = self.evaluate(
            self.eval_tasks
        )

        wandb.log(
                 {
                    'env_steps': self._n_env_steps_total,
                    'rollouts': self._n_rollouts_total,
                    'rl_steps': self._n_rl_update_steps_total,
                    'metrics/success_rate_eval': np.mean(success_rate_eval),
                    'metrics/return_eval_total': np.mean(np.sum(returns_eval, axis=-1)),
                    'metrics/total_steps_eval': np.mean(total_steps_eval),
                    'time_cost': (time.time() - self._start_time)/3600 + self._last_time,
                    'fps': (self._n_env_steps_total - self._n_env_steps_total_last)
                    / (time.time() - self._start_time_last)
                 },
                 step=self._n_env_steps_total
                 )
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()

        return np.mean(np.sum(returns_eval, axis=-1))

    def save_model(self, iter, perf, wandb_save=False):
        fname = f"agent_{iter}_perf{perf:.3f}.pt"
        save_path = os.path.join(
            logger.get_dir(), "save", fname
        )
        torch.save(self.agent.state_dict(), save_path)

        if wandb_save:
            logger.log(f"Save file {fname} to wandb")
            wandb.save(save_path)

    def load_model(self, ckpt_path):
        self.agent.load_state_dict(torch.load(ckpt_path,
                                   map_location=ptu.device))
        print("load successfully from", ckpt_path)
