from gym.envs.registration import register
import pdomains

register(
    "BlockPicking-Symm-v0",
    entry_point="pdomains.block_picking:BlockEnv",
    max_episode_steps=50,
)

register(
    "BlockPicking-Symm-RAD-v0",
    entry_point="pdomains.block_picking:BlockEnv",
    kwargs={"img_size": 90},
    max_episode_steps=50,
)

register(
    "BlockPulling-Symm-v0",
    entry_point="pdomains.block_pulling:BlockEnv",
    max_episode_steps=50,
)

register(
    "BlockPulling-Symm-RAD-v0",
    entry_point="pdomains.block_pulling:BlockEnv",
    kwargs={"img_size": 90},
    max_episode_steps=50,
)

register(
    "BlockPushing-Symm-v0",
    entry_point="pdomains.block_pushing:BlockEnv",
    max_episode_steps=50,
)

register(
    "BlockPushing-Symm-RAD-v0",
    entry_point="pdomains.block_pushing:BlockEnv",
    kwargs={"img_size": 90},
    max_episode_steps=50,
)

register(
    "DrawerOpening-Symm-v0",
    entry_point="pdomains.drawer_opening:DrawerEnv",
    max_episode_steps=50,
)

register(
    "DrawerOpening-Symm-RAD-v0",
    entry_point="pdomains.drawer_opening:DrawerEnv",
    kwargs={"img_size": 90},
    max_episode_steps=50,
)
