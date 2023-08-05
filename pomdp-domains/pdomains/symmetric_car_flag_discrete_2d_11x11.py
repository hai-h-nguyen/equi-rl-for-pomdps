# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import random
from tkinter import *
from tkinter import ttk
import time

IMAGE = 0
IMAGE_NORMED_FLATTEN = 1
VECTOR = 2

class CarEnv(gym.Env):
    def __init__(self, seed=0, rendering=False):

        self.visualize = rendering

        if self.visualize:
            # Create top-level window
            self.root = Tk()
            self.root.title("CarFlag 2D")

            # Create canvas to hold grid world
            self.canvas = Canvas(self.root, width="500", height="500")
            self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))

        # Create grid world
        self.grid_len = 11
        self.mid_len = self.grid_len // 2 - 1
        state_mat = np.ones((self.grid_len, self.grid_len))
        self.state_mat = state_mat

        # Determine pixel length of each block      
        num_col = state_mat.shape[1]
        pixel_width = 480
        while pixel_width % num_col != 0:
            pixel_width -= 1

        num_row = state_mat.shape[0]

        block_length = pixel_width / num_col

        self._state = 25

        if self.visualize:
            # Create rectangles
            for i in range(num_row):
                for j in range(num_col):
                    x_1 = self.grid_len + block_length * j
                    y_1 = self.grid_len + block_length * i
                    x_2 = x_1 + block_length
                    y_2 = y_1 + block_length

                    if self.state_mat[i][j] == 1:
                        color = "white"
                    else:
                        color = "black"

                    self.canvas.create_rectangle(x_1, y_1, x_2, y_2, fill=color)

        assert (self.grid_len - 1) % 2 == 0
        mid = self.grid_len//2
        self.info_coords = [[mid-1, mid], [mid, mid], [mid+1, mid],
                    [mid-1, mid-1], [mid, mid-1], [mid+1, mid-1],
                    [mid-1, mid+1], [mid, mid+1], [mid+1, mid+1]]
        self.info_cells = []
        for coord in self.info_coords:
            coord_x = coord[0]
            coord_y = coord[1]
            self.info_cells.append(coord_x * self.grid_len + coord_y)

        # color directional area
        if self.visualize:
            self._color(self.info_coords, 'blue')
            self.root.update()

        self.action_space = spaces.Discrete(4)

        self.low_state = np.array(
            [0.0, 0.0, -1.0, -1.0], dtype=np.float32
        )
        self.high_state = np.array(
            [1.0, 1.0, 1.0, 1.0], dtype=np.float32
        )

        self.img_size = (2, self.grid_len, self.grid_len)
        self.image_space = gym.spaces.Box(
            shape=self.img_size, low=0, high=1.0, dtype=np.float32
        )

        self.obs_type = IMAGE

        if self.obs_type == IMAGE_NORMED_FLATTEN:
            self.observation_space = gym.spaces.Box(
                shape=(np.array(self.img_size).prod(),), low=0, high=1.0, dtype=np.float32
            )
        elif self.obs_type == IMAGE:
            self.observation_space = gym.spaces.Box(
                shape=self.img_size, low=0, high=255.0, dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                shape=(4,), low=0, high=1.0, dtype=np.float32
            )

        self.goal_loc = None
        self.enter_info = False

        self.discount = 0.9
        self.step_reward = 0.0

    def _convert_coordinate(self, coord_x, coord_y):
        """Convert a (x, y) point to a 1D number

        Args:
            coord_x (int): x coordinate (from 0, horizontal)
            coord_y (int): y coordinate (from 0, vertical)
        """
        return coord_x * self.grid_len + coord_y + 1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """an environment step function

        Args:
            action (int): 0: left, 1: up, 2: right, 3: down
        """
        old_state = np.copy(self._state)
        old_state = int(old_state)

        done = False
        reward = self.step_reward
        enter_info_area = False

        self._state = self._get_next_state(self.state_mat, np.copy(old_state),action)

        # Clear the previous position
        if self._state != old_state and self.visualize:
            if old_state not in self.info_cells:
                self.canvas.itemconfig(old_state + 1, fill="white")
            else:
                self.canvas.itemconfig(old_state + 1, fill="blue")

        if self._state != self.goal_loc:
            if self.visualize:
                self.canvas.itemconfig(self._state + 1, fill="red")
        else:
            # Reached the goal
            if self.visualize:
                self.canvas.itemconfig(self._state + 1, fill="orange")
            done = True
            reward += 1

        if self._state in self.info_cells:
            # print("Get into info area")
            self.enter_info = True
            if self.visualize:
                self.canvas.itemconfig(self._state + 1, fill="yellow")
            enter_info_area = True
        else:
            if self.visualize:
                self._color(self.info_coords, 'blue')

        if self.visualize:
            self.root.update()

        obs = self._prepare_obs(self._state, self.goal_loc if enter_info_area else None)

        if self.visualize:
            time.sleep(0.1)

        info = {}
        info["success"] = (reward != self.step_reward)

        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def _check_far_enough(self, state, goal):
        """check if a state is 2 steps away from goal

        Args:
            state (int): _description_
            goal (int): _description_

        Returns:
            bool: true if far enough
        """
        state_xy = self._state_to_xy(state)
        goal_xy = self._state_to_xy(goal)

        dist = abs(state_xy[0] - goal_xy[0]) + abs(state_xy[1] - goal_xy[1])

        return dist >= 2


    def reset(self):
        # Clear the previous goal coors
        if self.goal_loc is not None and self.visualize:
            self.canvas.itemconfig(self.goal_loc + 1, fill="white")

        self.enter_info = False

        self.goal_loc = random.randint(0, self.grid_len**2 - 1)

        while self.goal_loc in self.info_cells:
            self.goal_loc = random.randint(0, self.grid_len**2 - 1)

        # Hightlight the chosen goal area
        if self.visualize:
            self.canvas.itemconfig(self.goal_loc + 1, fill="green")

        # Initialize the position such that it doesn't overlap w/ the goal
        # or the info coords or is not far enough from the goal
        while (self._state == self.goal_loc or
               self._state in self.info_cells or
               not self._check_far_enough(self._state, self.goal_loc)):
            self._state = random.randint(0, self.grid_len**2 - 1)

        if self.visualize:
            # Hightlight the agent
            self.canvas.itemconfig(self._state + 1, fill="red")

            # Highlight info area
            self._color(self.info_coords, 'blue')

            self.root.update()

        return self._prepare_obs(self._state)

    def _prepare_obs(self, state, goal=None):
        """observation is a two channel image
        channel 0: encodes the agent position
        channel 1: encodes the goal position

        Args:
            state (int): agent position
            goal (int, optional): goal position. Defaults to None.
        """

        if self.obs_type in [IMAGE_NORMED_FLATTEN, IMAGE]:
            obs = np.zeros((2, self.grid_len, self.grid_len))
        else:
            obs = np.zeros((self.grid_len, self.grid_len))
        x, y = self._state_to_xy(state)

        if self.obs_type == IMAGE:
            obs[0, y, x] = 255.0
        elif self.obs_type == IMAGE_NORMED_FLATTEN:
            obs[0, y, x] = 1.0

        if goal is not None:
            xg, yg = self._state_to_xy(self.goal_loc)
            if self.obs_type == IMAGE:
                obs[1, yg, xg] = 255.0
            elif self.obs_type == IMAGE_NORMED_FLATTEN:
                obs[1, yg, xg] = 1.0

        # for coord in self.info_coords:
        #     obs[0, coord[1], coord[0]] = 1.0
        if goal is None:
            xg, yg = 0.0, 0.0

        if self.obs_type == VECTOR:
            obs = np.array([x, y, xg, yg])/self.grid_len

        if self.obs_type == IMAGE_NORMED_FLATTEN:
            return obs.flatten()

        return obs


    def _state_to_xy(self, state):
        """convert a state to x y coordinate (the origin is in the top left corner)

        Args:
            state (int): index

        Returns:
            List: [x, y]
        """
        x = state % self.grid_len
        y = (state - x) // self.grid_len

        return [x, y] 

    def _convert_to_xy(self, x, y):
        """Convert (x, y) coordinates to symmetric ones
        by moving the origin from the top left corner to
        the center of the grid 

        Args:
            x (int): coordinate x in the old origin
            y (int): coordinate y in the old origin

        Returns:
            List: symmetric coordinates
        """
        x_sym, y_sym = 0.0, 0.0

        mid_grid_len = self.mid_len + 0.5

        # top quadrants
        if 0 <= y <= self.mid_len:
            x_sym = x - mid_grid_len
            y_sym = mid_grid_len - y

        # bottom quadrants
        if self.mid_len + 1 <= y <= self.grid_len - 1:
            x_sym = x - mid_grid_len
            y_sym = mid_grid_len - y

        assert x_sym != 0.0
        assert y_sym != 0.0

        return [x_sym, y_sym]

    def _color_goal_area(self, goal_loc, color='green'):
        """color goal area

        Args:
            goal_loc (List):
            color (str):
        """
        if self.visualize:
            self.canvas.itemconfig(self._convert_coordinate(goal_loc[0], goal_loc[1]), fill=color)

    def _get_next_state(self, state_mat, state, action):

        num_col = state_mat.shape[1]
        num_row = state_mat.shape[0]

        state_row = int(state/num_col)
        state_col = state % num_col

        # If action is "left"
        if action == 2:
            if state_col != 0 and state_mat[state_row][state_col - 1] == 1:
                # print("Moving Left")
                state -= 1
        # If action is "up"
        elif action == 1:
            if state_row != 0 and state_mat[state_row - 1][state_col] == 1:
                # print("Moving Up")
                state -= num_col
        # If action is "right"
        elif action == 0:
            if state_col != (num_col - 1) and state_mat[state_row][state_col + 1] == 1:
                # print("Moving Right")
                state += 1
        # If action is "down"
        else:
            if state_row != (num_row - 1) and state_mat[state_row + 1][state_col] == 1:
                # print("Moving Down")
                state += num_col

        return state

    def _color(self, coordinates, color):
        """Color given coordinates

        Args:
            coordinates (List): a list holding coordinates to color
            color (str): color to fill
        """
        for coord in coordinates:
            self.canvas.itemconfig(self._convert_coordinate(coord[0], coord[1]), fill=color)
