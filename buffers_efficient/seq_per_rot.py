import numpy as np
from utils.segment_tree import SumSegmentTree, MinSegmentTree
from buffers_efficient.seq_rot import SeqRotBuffer


class SeqPerRotBuffer(SeqRotBuffer):
    buffer_type = "seq_per_rot_eff"

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        sampled_seq_len: int,
        sample_weight_baseline: float,
        num_aug_episode: int,
        alpha=0.6,
        **kwargs,
    ):
        super().__init__(max_replay_buffer_size,
                         observation_dim,
                         action_dim,
                         sampled_seq_len,
                         sample_weight_baseline,
                         num_aug_episode,
                         **kwargs)

        # prioritized replay buffer
        self.beta = 0.6
        it_capacity = 1
        while it_capacity < max_replay_buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._episode_cnt = 0
        self._episode_tracker = {}

        assert alpha > 0
        self._alpha = alpha

    def update_priorities(self, episode_idxes, priorities):
        '''
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Args:
          - idxes: List of idxes of sampled transitions
          - priorities: List of updated priorities corresponding to
                        transitions at the sampled idxes denoted by
                        variable `idxes`.
        '''
        assert len(episode_idxes) == len(priorities)
        for idx, priority in zip(episode_idxes, priorities):

            if priority <= 0:
                print("Invalid priority:", priority)
                print("All priorities:", priorities)

            assert priority > 0
            assert 0 <= idx < self._episode_cnt
            self._it_sum[idx] = priority[0] ** self._alpha
            self._it_min[idx] = priority[0] ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def clear(self):
        self._top = 0  # trajectory level (first dim in 3D buffer)
        self._size = 0  # trajectory level (first dim in 3D buffer)
        self._episode_cnt = 0

    def _add_episode(self, observations, actions, rewards,
                     terminals, next_observations, expert_masks):
        """
        NOTE: must add one whole episode/sequence/trajectory,
                        not some partial transitions
        the length of different episode can vary, but must be greater than 2
                so that the end of valid_starts is 0.

        all the inputs have 2D shape of (L, dim)
        """

        seq_len = observations.shape[0]  # L
        indices = list(
            np.arange(self._top, self._top + seq_len) % self._max_replay_buffer_size
        )

        self._observations[indices] = observations
        self._actions[indices] = actions
        self._expert_masks[indices] = expert_masks
        self._rewards[indices] = rewards
        self._terminals[indices] = terminals
        self._ends[indices] = 0

        self._valid_starts[indices] = self._compute_valid_starts(seq_len)
        self._top = (self._top + seq_len) % self._max_replay_buffer_size

        # add final transition: obs is useful but the others are just padding
        self._observations[self._top] = next_observations[-1]  # final obs
        self._actions[self._top] = 0.0
        self._rewards[self._top] = 0.0
        self._terminals[self._top] = 1
        self._valid_starts[self._top] = 0.0  # never be sampled as starts
        self._expert_masks[self._top] = expert_masks[-1]
        self._ends[self._top] = 1  # the end of one episode

        self._top = (self._top + 1) % self._max_replay_buffer_size
        self._size = min(self._size + seq_len + 1, self._max_replay_buffer_size)

        self._it_sum[self._episode_cnt] = self._max_priority ** self._alpha
        self._it_min[self._episode_cnt] = self._max_priority ** self._alpha

        self._episode_tracker[self._episode_cnt] = self._top

        self._episode_cnt += 1

    def random_episodes(self, batch_size):
        """
        return each item has 3D shape (sampled_seq_len, batch_size, dim)
        """
        sampled_episode_starts, weights, episode_indices = self._sample_indices(batch_size)  # (B,)

        # get sequential indices
        indices = []
        next_indices = []  # for next obs
        for start in sampled_episode_starts:  # small loop
            end = start + self._sampled_seq_len  # continuous + T
            indices += list(np.arange(start, end) % self._max_replay_buffer_size)
            next_indices += list(
                np.arange(start + 1, end + 1) % self._max_replay_buffer_size
            )

        # extract data
        batch = self._sample_data(indices, next_indices)
        # each item has 2D shape (num_episodes * sampled_seq_len, dim)

        # generate masks (B, T)
        masks = self._generate_masks(indices, batch_size)
        batch["mask"] = masks

        for k in batch.keys():
            batch[k] = (
                batch[k]
                .reshape(batch_size, self._sampled_seq_len, -1)
                .transpose(1, 0, 2)
            )

        return batch, episode_indices, weights

    def _sample_indices(self, batch_size):
        # self._top points at the start of a new sequence
        # self._top - 1 is the end of the recently stored sequence
        # valid_starts_indices = np.where(self._valid_starts > 0.0)[0]

        episode_start_indices = []
        episode_indices = []
        weights = []

        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self._episode_cnt) ** (-self.beta)

        for _ in range(batch_size):
            # TODO: check range
            mass = np.random.rand() * self._it_sum.sum(0, self._episode_cnt - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            episode_indices.append(idx)
            episode_start_indices.append(self._episode_tracker[idx])

            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self._episode_cnt) ** (-self.beta)
            weights.append(weight / max_weight)
        weights = np.array(weights).reshape((-1, 1))

        return episode_start_indices, weights, episode_indices

    def get_state_dict(self):
        save_dict = {}

        # only save non-zero data
        save_dict["_observations"] = self._observations[:self._size]
        save_dict["_actions"] = self._actions[:self._size]
        save_dict["_expert_masks"] = self._expert_masks[:self._size]
        save_dict["_rewards"] = self._rewards[:self._size]
        save_dict["_terminals"] = self._terminals[:self._size]
        save_dict["_valid_starts"] = self._valid_starts[:self._size]
        save_dict["_top"] = self._top
        save_dict["_size"] = self._size
        save_dict["_ends"] = self._ends

        save_dict["_alpha"] = self._alpha
        save_dict["_it_sum"] = self._it_sum
        save_dict["_it_min"] = self._it_min
        save_dict["_max_priority"] = self._max_priority
        save_dict["_episode_tracker"] = self._episode_tracker
        save_dict["_episode_cnt"] = self._episode_cnt

        return save_dict

    def load_from_state_dict(self, state_dict):
        self._top = state_dict["_top"]
        self._size = state_dict["_size"]
        self._observations[:self._size] = state_dict["_observations"]
        self._actions[:self._size] = state_dict["_actions"]
        self._expert_masks[:self._size] = state_dict["_expert_masks"]
        self._rewards[:self._size] = state_dict["_rewards"]
        self._terminals[:self._size] = state_dict["_terminals"]
        self._valid_starts[:self._size] = state_dict["_valid_starts"]
        self._ends[:self._size] = state_dict["_ends"]

        self._alpha = state_dict["_alpha"]
        self._it_sum = state_dict["_it_sum"]
        self._it_min = state_dict["_it_min"]
        self._max_priority = state_dict["_max_priority"]
        self._episode_tracker = state_dict["_episode_tracker"]
        self._episode_cnt = state_dict["_episode_cnt"]


if __name__ == "__main__":
    buffer_size = 1000
    obs_dim = (2, 84, 84)
    act_dim = 5
    sampled_seq_len = 7
    baseline = 0.0
    num_aug_episode= 2
    buffer = SeqPerRotBuffer(
        buffer_size, obs_dim, act_dim, sampled_seq_len, baseline, num_aug_episode
    )
    for l in range(sampled_seq_len - 1, sampled_seq_len + 5):
        print(l)
        assert buffer._compute_valid_starts(l)[0] > 0.0
        print(buffer._compute_valid_starts(l))

    for _ in range(200):
        e = np.random.randint(3, 10)
        buffer.add_episode(
            np.random.rand(e, *obs_dim),
            np.zeros((e, act_dim)),
            np.zeros((e, 1)),
            np.zeros((e, 1)),
            np.random.rand(e, *obs_dim),
            np.ones((e, 1)),
        )
    print(buffer._size, buffer._top)

    for _ in range(10):
        batch, _, _ = buffer.random_episodes(1)  # (T, B, dim)
        print(batch["obs"][:, 0, 0])
        print(batch["obs2"][:, 0, 0])
        print(batch["mask"][:, 0, 0].astype(np.int32))
        print(batch["term"][:, 0, 0])
        print("\n")
