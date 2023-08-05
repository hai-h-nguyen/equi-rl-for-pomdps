from buffers_efficient.seq_vanilla import SeqBuffer
from utils.helpers import get_random_transform_params, perturb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SeqRotBuffer(SeqBuffer):
    buffer_type = "seq_rot_eff"

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        sampled_seq_len: int,
        sample_weight_baseline: float,
        num_aug_episode: int,
        **kwargs
    ):

        super().__init__(max_replay_buffer_size,
                         observation_dim,
                         action_dim,
                         sampled_seq_len,
                         sample_weight_baseline,
                         **kwargs)

        self._num_aug_ep = num_aug_episode
        self._save_image = False

    def add_episode(self, observations, actions, rewards,
                    terminals, next_observations, expert_masks):
        
        if observations.shape[0] >= 2:

            assert (
                observations.shape[0]
                == actions.shape[0]
                == rewards.shape[0]
                == terminals.shape[0]
                == next_observations.shape[0]
                == expert_masks.shape[0]
            )

            self._add_episode(observations, actions, rewards,
                              terminals, next_observations, expert_masks)
            self._augment_and_add_episodes(observations, actions, rewards,
                                           terminals, next_observations,
                                           expert_masks)
            return True
        else:
            return False

    def _augment_and_add_episodes(self, observations, actions, rewards,
                                  terminals, next_observations, expert_masks):
        """Augment episode"""
        seq_len = observations.shape[0]
        image_size = self._observation_dim[1:]

        for ep_idx in range(self._num_aug_ep):
            new_observations = observations.copy()
            new_actions = actions.copy()
            new_next_observations = next_observations.copy()

            # Compute random rigid transform.
            # Same for the entire history
            theta, trans, pivot = get_random_transform_params(image_size)

            for idx in range(seq_len):
                if self._save_image:
                    plt.imshow(observations[idx][0])
                    plt.clim(0, 0.3)
                    plt.colorbar()
                    plt.savefig(f"before_{ep_idx}_{idx}.png", bbox_inches='tight')
                    plt.close()

                obs, next_obs, dxy, _ = perturb(observations[idx][0].copy(),
                                                next_observations[idx][0].copy(),
                                                actions[idx][1:3],
                                                theta, trans, pivot,
                                                set_trans_zero=True)
                if self._save_image:
                    plt.imshow(obs)
                    plt.clim(0, 0.3)
                    plt.colorbar()
                    plt.savefig(f"after_{ep_idx}_{idx}.png", bbox_inches='tight')
                    plt.close()

                new_observations[idx][0] = obs
                new_actions[idx][1:3] = dxy
                new_next_observations[idx][0] = next_obs

            self._add_episode(new_observations, new_actions, rewards,
                              terminals, new_next_observations, expert_masks)

if __name__ == "__main__":
    buffer_size = 100
    obs_dim = (2, 84, 84)
    act_dim = 5
    sampled_seq_len = 7
    baseline = 0.0
    num_aug_episode= 2
    buffer = AugSeqReplayBufferEfficient(
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
