import os, sys, tensorflow.keras, random

from tools.utils.ai_lab_functions import rolling

module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path:
    sys.path.append(module_path)

import gym, envs
from utils.ai_lab_functions import *
from timeit import default_timer as timer
from tqdm import tqdm as tqdm
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_model(input_size, output_size, hidden_layer_size, hidden_layer_number):
    """
    Args:
        input_size: the number of nodes for the input layer
        output_size: the number of nodes for the output layer
        hidden_layer_size: the number of nodes for each hidden layer
        hidden_layer_number: the number of hidden layers
    """
    model = Sequential()
    # Input layer + first hidden layer
    model.add(Dense(hidden_layer_size, input_dim=input_size, activation='relu'))
    # Other hidden layers
    for _ in range(hidden_layer_number-1):
        model.add(Dense(hidden_layer_size, activation='relu'))
    # Output layer = the input is only repoted
    model.add(Dense(output_size, activation='linear'))
    # Creating model and define loss function (= funzione da minimizzare) and optimizer (= algoritmo di ottimizzazione)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_model(model, memory, batch_size, gamma=0.99):
    """
    Performs the value iteration algorithm for a specific environment
    Args:
        model: the neural network model to train
        memory: the memory array on wich perform the training
        batch_size: the size of the batch sampled from the memory
        gamma: gamma value, the discount factor for the Bellman equation
    """
    if len(memory)< batch_size:
        return model
    # Sample a random batch from the memory
    batch = random.sample(memory, batch_size)
    print(len(batch))
    print("Start training")
    cont = 0
    for (s, a, s_1, r, final_state) in batch:
        s = s.reshape(1, 4)
        target = model.predict(s)[0]
        if final_state:
            target[a] = r
        else:
            max_q = np.argmax(model.predict(s_1.reshape(1, 4))[0])
            target[a] = r + gamma * max_q
        #print(s, final_state)
        model.fit(s, np.array([target]), epochs=1, verbose=0)
        #print(target)
        cont+=1
        print(cont)
    print("Model trained")
    return model

def DQN(env, neural_network, trials, goal_score, batch_size, epsilon_decay=0.9995):
    """
    Args:
        env: OpenAI Gym environment
        neural_network: the neural network to train
        trials: the number of iterations for the training phase
        goal_score: the minimum score to consider the problem 'solved'
        batch_size: the size of the batch sampled from the memory
        epsilon_decay: the decay value of epsilon for the eps-greedy exploration
    """

    epsilon = 1.0
    epsilon_min = 0.01

    score_queue = [] # coda di score che dice nei vari traial quanto è andata bene
    experience = deque(maxlen=10000)
    r = 0
    # fit modifica i passi in maniera tale da correggere l'input sull'output desiderato
    for trial in range(trials):
        s = env.reset()
        final_state = False
        score = 0
        print("Trial: ", trial)
        while not final_state:
            # Epsilon greedy exploration
            if random.random() < epsilon:
                a = env.action_space.sample()
            else:
                s = s.reshape(1, 4)
                a = np.argmax(neural_network.predict(s))

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            # Step
            s_1, r, final_state, _ = env.step(a)
            experience.append((np.array(s), a, np.array(s_1), r, final_state))
            train_model(neural_network, experience, batch_size)
            s = s_1
            score+= r
        score_queue.append(score)
        print("Episode: {:7.0f}, Score: {:3.0f}, EPS: {:3.2f}".format(trial, score_queue[-1], epsilon))
        if (score > goal_score): break

    return neural_network, score_queue

if __name__=="__main__":
    # replay_buffer = deque(maxlen=100000)
    # for _ in range(100): replay_buffer.append(random.uniform(0,1))
    # samples = random.sample(replay_buffer, 3)
    rewser = []
    window = 10

    env = gym.make("CartPole-v1")
    neural_network = create_model(4, 2, 32, 2)
    neural_network, score = DQN(env, neural_network, trials=1000, goal_score=130, batch_size=1)

    score = rolling(np.array(score), window)
    rewser.append({"x": np.arange(1, len(score) + 1), "y": score, "ls": "-", "label": "DQN"})
    plot(rewser, "Rewards", "Episodes", "Rewards")