from pdomains import *
import matplotlib.pyplot as plt
import gym
import time

env=gym.make('pdomains-car-flag-symm-2d-p2-v0', rendering=True)
obs = env.reset()
time.sleep(1)

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(0.5)
    # print(reward)
    if done:
        if "success" in info and info["success"] == True:
            print("Success")
        env.reset()
