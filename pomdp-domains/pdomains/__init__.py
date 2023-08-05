from gym.envs.registration import register

register(
    id='pdomains-car-flag-symm-v1',
    entry_point='pdomains.symmetric_car_flag_discrete_1d:CarEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-car-flag-symm-v2',
    entry_point='pdomains.symmetric_car_flag_discrete_1d_25:CarEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-car-flag-n5-v0',
    entry_point='pdomains.symmetric_car_flag_discrete_1d:CarEnv',
    max_episode_steps=50,
    kwargs={"priest_pos": -5.0}
)

register(
    id='pdomains-car-flag-p5-v0',
    entry_point='pdomains.symmetric_car_flag_discrete_1d:CarEnv',
    max_episode_steps=50,
    kwargs={"priest_pos": 5.0}
)

register(
    id='pdomains-car-flag-n10-v0',
    entry_point='pdomains.symmetric_car_flag_discrete_1d:CarEnv',
    max_episode_steps=50,
    kwargs={"priest_pos": -10.0}
)

register(
    id='pdomains-car-flag-p10-v0',
    entry_point='pdomains.symmetric_car_flag_discrete_1d:CarEnv',
    max_episode_steps=50,
    kwargs={"priest_pos": 10.0}
)

register(
    id='pdomains-car-flag-symm-continuous-v0',
    entry_point='pdomains.symmetric_car_flag_continuous_1d:CarEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-car-flag-symm-2d-v0',
    entry_point='pdomains.symmetric_car_flag_discrete_2d:CarEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-car-flag-symm-2d-dreamer-v0',
    entry_point='pdomains.symmetric_car_flag_discrete_2d_dreamer:CarEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-car-flag-symm-2d-dreamer-v1',
    entry_point='pdomains.symmetric_car_flag_discrete_2d_11x11_dreamer:CarEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-car-flag-symm-2d-p1-v0',
    entry_point='pdomains.symmetric_car_flag_discrete_2d:CarEnv',
    max_episode_steps=50,
    kwargs={"info_offset": 1}
)

register(
    id='pdomains-car-flag-symm-2d-n1-v0',
    entry_point='pdomains.symmetric_car_flag_discrete_2d:CarEnv',
    max_episode_steps=50,
    kwargs={"info_offset": -1}
)

register(
    id='pdomains-car-flag-symm-2d-p2-v0',
    entry_point='pdomains.symmetric_car_flag_discrete_2d:CarEnv',
    max_episode_steps=50,
    kwargs={"info_offset": 2}
)

register(
    id='pdomains-car-flag-symm-2d-n2-v0',
    entry_point='pdomains.symmetric_car_flag_discrete_2d:CarEnv',
    max_episode_steps=50,
    kwargs={"info_offset": -2}
)

register(
    id='pdomains-car-flag-symm-2d-v1',
    entry_point='pdomains.symmetric_car_flag_discrete_2d_11x11:CarEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-block-picking-v0',
    entry_point='pdomains.block_picking:BlockEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-block-picking-pixel-v0',
    entry_point='pdomains.block_picking_pix:BlockEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-block-pulling-v0',
    entry_point='pdomains.block_pulling:BlockEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-block-pulling-pixel-v0',
    entry_point='pdomains.block_pulling_pix:BlockEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-block-pushing-v0',
    entry_point='pdomains.block_pushing:BlockEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-block-pushing-pixel-v0',
    entry_point='pdomains.block_pushing_pix:BlockEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-block-stacking-v0',
    entry_point='pdomains.block_stacking:BlockEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-drawer-opening-v0',
    entry_point='pdomains.drawer_opening:DrawerEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-drawer-opening-pixel-v0',
    entry_point='pdomains.drawer_opening_pix:DrawerEnv',
    max_episode_steps=50,
)

register(
    id='pdomains-drawer-opening-hard-v0',
    entry_point='pdomains.drawer_opening_hard:DrawerEnv',
    max_episode_steps=60,
)