from buffers.seq_rot import SeqRotBuffer
from utils.helpers import crop_observations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SeqRadRotBuffer(SeqRotBuffer):
    buffer_type = "seq_rad_rot"

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
                         num_aug_episode,
                         **kwargs)

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

            # crop first
            cropped_observations, cropped_n_observations = crop_observations(observations.copy(), next_observations.copy())

            # for idx in range(4):
            #     plt.imshow(observations[idx][0])
            #     plt.clim(0, 0.3)
            #     plt.colorbar()
            #     plt.savefig(f"rad_crop_before_{idx}.png", bbox_inches='tight')
            #     plt.close()

            #     plt.imshow(cropped_observations[idx][0])
            #     plt.clim(0, 0.3)
            #     plt.colorbar()
            #     plt.savefig(f"rad_crop_after_{idx}.png", bbox_inches='tight')
            #     plt.close()

            # breakpoint()

            # add cropped episodes
            self._add_episode(cropped_observations, actions, rewards,
                              terminals, cropped_n_observations, expert_masks)

            # finally, rotational augmentation
            self._augment_and_add_episodes(cropped_observations, actions,
                                           rewards, terminals,
                                           cropped_n_observations,
                                           expert_masks)
            return True
        else:
            return False


if __name__ == "__main__":
    seq_len = 10
    baseline = 1.0
    buffer = RadSeqRotBuffer(100, 2, 2, seq_len, baseline, 4)
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

