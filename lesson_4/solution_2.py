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

def policy_evaluation(env, policy, utility, discount=0.9, maxiters=300):
    """
    Update the current policy with che current value function
    """
    for i in range(maxiters):
        for state in range(env.observation_space.n):
            summ = 0
            for next_state in range(env.observation_space.n):
                summ += env.T[state,policy[state],next_state] * utility[next_state]
            utility[state] = env.RS[state] + summ*discount
    return utility


def value_iteration(env, maxiters=300, discount=0.9, max_error=1e-3):
    """
    Performs the value iteration algorithm for a specific environment

    Args:
        environment: OpenAI Gym environment
        maxiters: timeout for the iterations
        discount: gamma value, the discount factor for the Bellman equation
        max_error: the maximum error allowd in the utility of any state

    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
    """

    U_1 = [0 for _ in range(env.observation_space.n)]  # vector of utilities for states S
    delta = 0  # maximum change in the utility o any state in an iteration
    U = U_1.copy()


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
    #policy = value_iteration(env)

    print("\nTIME: \n{}".format(round(timer() - t, 4)))
    print("\nPOLICY:")
    #print(np.vectorize(env.actions.get)(policy.reshape(env.rows, env.cols)
