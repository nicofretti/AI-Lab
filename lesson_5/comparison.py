import os, sys

import numpy as np

module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path:
    sys.path.append(module_path)

import gym, envs
from utils.ai_lab_functions import *
from timeit import default_timer as timer
from tqdm import tqdm as tqdm

from solution_1 import q_learning, epsilon_greedy, softmax
from solution_2 import sarsa

if __name__=="__main__":
    envname = "Cliff-v0"

    print("\n----------------------------------------------------------------")
    print("\tEnvironment: ", envname)
    print("----------------------------------------------------------------\n")

    env = gym.make(envname)
    env.render()

    # Learning parameters
    episodes = 1000;ep_limit = 50;alpha = .3;gamma = .9;epsilon = .1;delta = 1e-3

    rewser = [];lenser = []

    window = 50  # Rolling window
    mrew = np.zeros(episodes)
    mlen = np.zeros(episodes)

    t = timer()

    # Q-Learning
    _, rews, lengths = q_learning(env, episodes, alpha, gamma, epsilon_greedy, epsilon)
    rews = rolling(rews, window)
    rewser.append({"x": np.arange(1, len(rews) + 1), "y": rews, "ls": "-", "label": "Q-Learning"})
    lengths = rolling(lengths, window)
    lenser.append({"x": np.arange(1, len(lengths) + 1), "y": lengths, "ls": "-", "label": "Q-Learning"})

    # SARSA
    _, rews, lengths = sarsa(env, episodes, alpha, gamma, epsilon_greedy, epsilon)
    rews = rolling(rews, window)
    rewser.append({"x": np.arange(1, len(rews) + 1), "y": rews, "label": "SARSA"})
    lengths = rolling(lengths, window)
    lenser.append({"x": np.arange(1, len(lengths) + 1), "y": lengths, "label": "SARSA"})

    print("Execution time: {0}s".format(round(timer() - t, 4)))

    plot(rewser, "Rewards", "Episodes", "Rewards")
    plot(lenser, "Lengths", "Episodes", "Lengths")