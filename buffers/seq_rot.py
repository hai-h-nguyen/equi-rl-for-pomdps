from buffers.seq_vanilla import SeqBuffer
from utils.helpers import get_random_transform_params, perturb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SeqRotBuffer(SeqBuffer):
    buffer_type = "seq_rot"

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
    seq_len = 10
    baseline = 1.0
    buffer = AugSeqReplayBuffer(100, 2, 2, seq_len, baseline)
    for l in range(seq_len - 2, seq_len + 2):
        print(l)
        assert buffer._compute_valid_starts(l)[0] > 0.0
        print(buffer._compute_valid_starts(l))
    for e in range(10):
        data = np.zeros((10, 2))
        buffer.add_episode(
            np.zeros((11, 2)),
            np.zeros((11, 2)),
            np.zeros((11, 1)),
            np.zeros((11, 1)),
            np.zeros((11, 2)),
        )
    print(buffer._size, buffer._valid_starts)
