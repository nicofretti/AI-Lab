import os
import sys

import numpy as np

module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path:
    sys.path.append(module_path)

import gym
from utils.ai_lab_functions import *
from timeit import default_timer as timer

def epsilon_greedy(q, state, epsilon):
    """
    Epsilon-greedy action selection function
    Args:
        q: q table
        state: agent's current state
        epsilon: epsilon parameter
    Returns:
        action id
    """
    if np.random.random() < epsilon:
        return np.random.choice(q.shape[1])
    return q[state].argmax()

def softmax(q, state, temp):
    """
    Softmax action selection function
    Args:
    q: q table
    state: agent's current state
    temp: temperature parameter
    Returns:
        action id
    """
    e = np.exp(q[state] / temp)
    return np.random.choice(q.shape[1], p=e / e.sum())

def q_learning(env, episodes, alpha, gamma, expl_func, expl_param):
    """
    Performs the Q-Learning algorithm for a specific environment
    Args:
        environment: OpenAI Gym environment
        episodes: number of episodes for training
        alpha: alpha parameter
        gamma: gamma parameter
        expl_func: exploration function (epsilon_greedy, softmax)
        expl_param: exploration parameter (epsilon, T)
    Returns:
        (policy, rewards, lengths): final policy, rewards for each episode [array], length of each episode [array]
    """

    Q = np.zeros((env.observation_space.n, env.action_space.n))  # Q(s, a)
    rews = np.zeros(episodes)
    lengths = np.zeros(episodes)

    for i in range(episodes):
        s = env.reset();end_state = False
        while not end_state:
            a = expl_func(Q, s, expl_param)
            s_1, r, end_state, _ = env.step(a)
            arr = []

            for a_1 in range(env.action_space.n):
                arr.append(Q[s_1][a_1])
            Q[s, a] = Q[s, a] + alpha * (r + gamma * max(arr) - Q[s, a])
            s = s_1

            rews[i] += r
            lengths[i] += 1

    policy = Q.argmax(axis=1)  # q.argmax(axis=1) automatically extract the policy from the q table
    return policy, rews, lengths


if __name__ == "__main__":
    envname = "Cliff-v0"

    print("\n----------------------------------------------------------------")
    print("\tEnvironment: {} \n\tQ-Learning".format(envname))
    print("----------------------------------------------------------------\n")

    env = gym.make(envname)
    env.render()
    print()

    # Learning parameters
    episodes = 500;alpha = .3;gamma = .9;epsilon = .1
    t = timer()

    # Q-Learning epsilon greedy
    policy, rewards, lengths = q_learning(env, episodes, alpha, gamma, epsilon_greedy, epsilon)
    print("Execution time: {0}s\nPolicy:\n{1}\n".format(round(timer() - t, 4),
                                                        np.vectorize(env.actions.get)(policy.reshape(
                                                            env.shape))))
    _ = run_episode(env, policy, 20)
