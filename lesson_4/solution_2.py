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

def policy_evaluation(env, policy, U, discount=0.9, maxiters=300):
    """
    Update the current policy with che current value function
    """
    for i in range(maxiters):
        for s in range(env.observation_space.n):
            summ = 0
            for s_1 in range(env.observation_space.n):
                summ += env.T[s, policy[s], s_1] * U[s_1]
            U[s] = env.RS[s] + discount * summ
    return U


def policy_iteration(environment, maxiters=150, discount=0.9, maxviter=10):
    """
    Performs the policy iteration algorithm for a specific environment

    Args:
        environment: OpenAI Gym environment
        maxiters: timeout for the iterations
        discount: gamma value, the discount factor for the Bellman equation

    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
    """

    policy = [0 for _ in range(environment.observation_space.n)]  # initial policy
    U = [0 for _ in range(environment.observation_space.n)]  # utility array

    unchanged = False
    iters = 0
    # Step (2) Policy Improvement
    while not unchanged and iters < maxiters:
        U = policy_evaluation(env, policy, U, discount, maxiters)
        unchanged = True
        for s in range(env.observation_space.n):
            max_policy = np.inf * -1
            arg_max_policy = -1
            for a in range(env.action_space.n):
                summ = 0
                for s_1 in range(env.observation_space.n):
                    summ += env.T[s, a, s_1] * U[s_1]
                if(summ > max_policy):
                    max_policy = summ
                    arg_max_policy = a

            summaries_utilty = 0
            for s_1 in range(env.observation_space.n):
                summaries_utilty += env.T[s, policy[s], s_1] * U[s_1]
            if (max_policy > summaries_utilty):
                policy[s] = arg_max_policy
                unchanged = False
        iters += 1
    return np.asarray(policy)

if __name__=="__main__":
    envname = "VeryBadLavaFloor-v0"

    print("\n----------------------------------------------------------------")
    print("\tEnvironment: {} \n\tPolicy Iteration".format(envname))
    print("----------------------------------------------------------------")

    env = gym.make(envname)
    print("\nRENDER:")
    env.render()

    t = timer()
    policy = policy_iteration(env)

    print("\nTIME: \n{}".format(round(timer() - t, 4)))
    print("\nPOLICY:")
    print(np.vectorize(env.actions.get)(policy.reshape(env.rows, env.cols)))