import os, sys, random

import numpy as np

from tools.utils.ai_lab_functions import values_to_policy

module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path:
    sys.path.append(module_path)

import gym, envs
from utils.ai_lab_functions import *
from timeit import default_timer as timer
from tqdm import tqdm as tqdm


"""
Value Iteration
In this method, the optimal policy is obained by choosing the action that
maximizes optimal state value function The optimal state value function is
obtained by iteratively solving the Bellman equation.
"""
def value_iteration(env, maxiters=300, discount=0.9, max_error=1e-3):
    """
    Args:
        env: OpenAI Gym environment
        maxiters: timeout for the iterations
        discount: gamma value, the discount factor for the Bellman equation
        max_error: the maximum error allowd in the utility of any state
    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
    """

    U_1 = [0 for _ in range(env.observation_space.n)]  # vector of utilities for states S
    error = max_error * (1-discount) / discount  # maximum error allowed in the utility of any state
    delta = 1
    U = U_1.copy()
    iters = 0
    while delta >= error and iters < maxiters:
        U = U_1.copy()
        delta = 0
        for s in range(env.observation_space.n):
            summatories = []
            for action in range(env.action_space.n):
                summ = 0
                for s_1 in range(env.observation_space.n):
                    summ += env.T[s, action, s_1] * U[s_1]
                summatories.append(summ)
            U_1[s] = env.RS[s] + discount * max(summatories)
            delta = max(abs(U_1[s]-U[s]),delta)
        iters += 1
    return values_to_policy(np.asarray(U), env)  # automatically convert the value matrix U to a policy


if __name__=="__main__":
    envname = "LavaFloor-v0"

    print("\n----------------------------------------------------------------")
    print("\tEnvironment: {} \n\tValue Iteration".format(envname))
    print("----------------------------------------------------------------")

    env = gym.make(envname)
    print("\nRENDER:")
    env.render()

    t = timer()
    print(env)
    policy = value_iteration(env)

    print("\nTIME: \n{}".format(round(timer() - t, 4)))
    print("\nPOLICY:")
    print(np.vectorize(env.actions.get)(policy.reshape(env.rows, env.cols)))
    print("\n----------------------------------------------------------------")

