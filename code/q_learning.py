"""
Training and evaluating the q-learning agent on the LunarRoverEnv
"""
import argparse
from lunar_rover_env import LunarRoverEnv
from collections import defaultdict
import csv
import os
from itertools import zip_longest
import numpy as np

"""
Turning Minigrid's observation into dict key
"""
def observationToState(observation):
    image = observation["image"]
    dir = int(observation["direction"])
    obs_key = (image.tobytes(), image.shape, dir)
    return obs_key

"""
Encoding our next action to an integer to input
into the environment
"""
def chooseAction(q_table, state, epsilon, action_count, rng)->int:
    if rng.random() < epsilon:
        return int(rng.integers(action_count))
        
    return int(np.argmax(q_table[state]))

"""
Initialize Q-table and return as a dict
"""
def createQTable(num_actions:int):
    return defaultdict(lambda: np.zeros(num_actions, dtype=np.float64))

"""
Linear Epsilon Decay based off total number of episodes, current episode,
end goal epsilon and starting epsilon
- this will encourage more exploration in earlier epsiodes and greedy choices
in later episodes
"""
def epsilonDecay(end_eps:float, start_eps:float, tot_episode:int, episode:int)->float:
    frac = episode / max(tot_episode - 1, 1)
    new_epsilon = start_eps + frac * (end_eps - start_eps)
    return new_epsilon


"""
Calculates value to update Q-table
"""
def updateQTable(terminated, q_table, state, next_state, action, reward, gamma, alpha):
    next_val = 0.0 if terminated else np.max(q_table[next_state])
    td_target = reward + gamma * next_val
    updated_val = alpha * (td_target - q_table[state][action])
    return updated_val
