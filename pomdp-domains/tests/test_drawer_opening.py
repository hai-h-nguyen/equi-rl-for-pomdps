from pdomains import *
import matplotlib.pyplot as plt
import gym
import matplotlib
matplotlib.use('Agg')

env=gym.make('pdomains-drawer-opening-hard-v0', rendering=True)
obs = env.reset()

# plt.imshow(obs[0])
# plt.clim(0, 0.3)
# plt.colorbar()
# plt.savefig('test.png', bbox_inches='tight')
# breakpoint()

cnt = 0
ep_idx = 0
success_ep_idx = 0
ret = 0

while ep_idx <= 10:
    # action = env.action_space.sample()
    action = env.query_expert(ep_idx)
    obs, reward, done, info = env.step(action)
    cnt += 1
    ret += reward
    if done:
        if info["success"]:
            success_ep_idx += 1
        # print(cnt, success_ep_idx, ep_idx)
        print(f"ret: {ret:.2f} {cnt}")
        ret = 0
        ep_idx += 1
        cnt = 0
        env.reset()
