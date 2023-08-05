from pdomains import *
import matplotlib.pyplot as plt
import gym
import argparse


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--num_episodes', type=int, default=100, help='number of expert episodes (default 100)')
parser.add_argument('--ratio', type=float, default=0.5, help='pick correct/incorrect ratio (default 0.5)')
parser.add_argument('--name', type=str, default='expert.', help='name of file (default expert.)')

args = parser.parse_args()

env = gym.make('pdomains-block-picking-v0', rendering=True)
obs = env.reset()

epi_cnt = 0

pick_correct_ep = (int) (args.ratio*args.num_episodes)
pick_incorrect_ep = args.num_episodes - pick_correct_ep

# while epi_cnt < pick_correct_ep:
#     action = env.pick_movable()
#     obs, reward, done, info = env.step(action)
#     if done:
#         epi_cnt += 1
#         env.reset()

step_cnt = 0
epi_cnt = 0
while epi_cnt < pick_incorrect_ep:
    if step_cnt < 10:
        action = env.pick_immovable()
    elif step_cnt < 15:
        action = env.move_up()
    else:
        action = env.pick_movable()
    obs, reward, done, info = env.step(action)
    step_cnt += 1
    if done:
        epi_cnt += 1
        step_cnt = 0
        env.reset()
