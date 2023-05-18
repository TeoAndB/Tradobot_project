from __future__ import annotations
import time


import numpy as np
import pandas as pd
import random
from collections import deque, namedtuple


# PyTorch packages
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn


# configuration files
from src.config_model import *
from src.config_data import *

# environment
from src.environments.FinRL_environment import *
from src.features.preprocessor_FinRL import data_split

# omg I need pseudocode
import torch
import torch.nn as nn
import numpy as np



TECH_INDICATORS = INDICATORS
INDEX = DATASET_INDEX - 1

# ARGUMENTS IN train_model_py
# TODO: delete and move class call in train_model.py
processed = pd.read_csv(f'{input_filepath}/{TRAIN_DATASET}')
processed.rename(columns={'timestamp': 'date'}, inplace=True)
print(processed)
stock_dimension = len(processed.tic.unique())
state_space = 1 + 2 * stock_dimension + len(TECH_INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
    "hmax": hmax,
    "initial_amount": initial_amount,
    "buy_cost_pct": buy_cost_pct,
    "sell_cost_pct": sell_cost_pct,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": reward_scaling,
    "print_verbosity": print_verbosity
}

# PARAMS FOR MODEL
# TODO: add to model_config.py
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size

TAU = 1e-3  # for soft update of target parameters
LR = 1e-3  # learning rate
UPDATE_EVERY = 5  # how often to update the network

epsilon_decay = 0.995 # decrease exploration rate as the agent becomes good at trading

gamma = 0.95
epsilon = 1.0  # initial exploration rate
epsilon_min = 0.01  # minimum exploration rate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Portfolio:
    def __init__(self, balance=initial_amount):
        self.initial_portfolio_value = balance
        self.balance = balance
        self.initial_balance = balance
        self.inventory = []
        self.return_rates = []
        self.portfolio_values = [balance]
        self.buy_dates = []
        self.sell_dates = []

    def reset_portfolio(self):
        self.balance = self.initial_portfolio_value
        self.inventory = []
        self.return_rates = []
        self.portfolio_values = [self.initial_portfolio_value]


import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBranchNet(nn.Module):
    def __init__(self, num_branches, r, c):
        super(MultiBranchNet, self).__init__()
        self.num_branches = num_branches
        self.r = r
        self.c = c

        # Define the shared layers of the network
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Define the branches of the network
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            branch = nn.Sequential(
                nn.Linear(16 * ((r - 2) // 2) * ((c - 2) // 2), 64),
                nn.ReLU(),
                nn.Linear(64, 12)
            )
            self.branches.append(branch)

    def forward(self, x):
        # Reshape the input to have a single channel dimension
        x = x.view(-1, 1, self.r, self.c)

        # Apply the shared layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = x.view(-1, 16 * ((self.r - 2) // 2) * ((self.c - 2) // 2))

        # Feed the output to each output branch
        outputs = []
        for i in range(self.num_branches):
            outputs.append(self.branches[i](x))

        #  final output of the network is a tuple containing 3 tensors, each of size (batch_size, 12)
        return tuple(outputs)


# net = MultiBranchNet(num_branches=3, r=28, c=28)

class Agent(Portfolio):
    def __init__(self, action_dim, state_dim, balance, is_eval=False):
        super().__init__(balance=balance)
        self.model_type = 'DQN'
        self.state_dim = state_dim
        self.action_dim = action_dim  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 60
        self.batch_size = 60

        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995 # decrease exploration rate as the agent becomes good at trading
        self.is_eval = is_eval
        self.model = MultiBranchNet(num_branches=3, r=state_dim, c=1) if not is_eval else None

        # self.tensorboard = TensorBoard(log_dir='./logs/DQN_tensorboard', update_freq=90)
        # self.tensorboard.set_model(self.model)

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0 # reset exploration rate

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        options = self.model(state)

        # returns (action_stock1, action_stock2, action_stock3)
        return tuple(np.argmax(o) for o in options)

    def expReplay(self):
        # retrieve recent buffer_size long memory
        mini_batch = [self.memory[i] for i in range(len(self.memory) - self.buffer_size + 1, len(self.memory))]

        for state, actions, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model(next_state)[0])

            target_f = self.model(state) #



    def experience_replay(self):
        # retrieve recent buffer_size long memory
        mini_batch = [self.memory[i] for i in range(len(self.memory) - self.buffer_size + 1, len(self.memory))]

        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                Q_target_values = [] # target values V for each stock
                for i in range(self.action_dim):
                    # create a copy of the state and set the i-th element to 1, representing the action
                    state_i = np.copy(state) # state size = stock * features
                    state_i[:, i] = 1
                    Q_target_values.append(reward + self.gamma * np.amax(self.model(state_i)[i]))
            else:
                Q_target_values = [reward] * self.action_dim

            # update the Q-values for the selected actions
            Q_target_values = np.asarray(Q_target_values).reshape(1, -1)
            actions = actions.reshape(1, -1)
            next_actions = self.model(state)
            for i in range(self.action_dim):
                next_actions[i][actions[0][i]] = Q_target_values[0][i]
            next_actions = tuple(next_actions)
            self.model.zero_grad()
            loss_fn = nn.MSELoss()
            loss = sum([loss_fn(next_actions[i], self.model(state)[i]) for i in range(self.action_dim)])
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()