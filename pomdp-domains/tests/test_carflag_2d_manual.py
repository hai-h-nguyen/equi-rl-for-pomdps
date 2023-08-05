from pdomains import *
import matplotlib.pyplot as plt
import gym
from getkey import getkey, keys
import numpy as np


env=gym.make('pdomains-car-flag-symm-2d-dreamer-v1', rendering=True)
obs = env.reset()
print(obs)

for _ in range(100):

    while True:

        key = getkey()

        if key == keys.LEFT:
            action = 2
            break

        if key == keys.RIGHT:
            action = 0
            break

        if key == keys.UP:
            action = 1
            break

        if key == keys.DOWN:
            action = 3
            break

    obs, reward, done, info = env.step(action)
    print(obs)
    print(reward, done, info)

    if done:
        env.reset()