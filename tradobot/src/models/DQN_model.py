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
from tensorflow.keras.models import load_model
import torch.nn as nn

# other packages
# import glob
# import io
# import base64
# from IPython.display import HTML
# from IPython import display as ipythondisplay
# from pyvirtualdisplay import Display
# from collections import deque, namedtuple


def show_video(folder):
    mp4list = glob.glob('%s/*.mp4' % folder)
    if len(mp4list) > 0:
        encoded = base64.b64encode(io.open(mp4list[0], 'r+b').read())
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay loop controls style="height: 400px;"> 
        <source src="data:video/mp4;base64,{0}" type="video/mp4" /> </video>'''.format(encoded.decode('ascii'))))


display = Display(visible=0, size=(400, 300))
display.start()


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
    def __init__(self, balance=50000):
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



class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_space, action_size, seed, fc1_units=256, fc2_units=256):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_space = state_space
        self.action_size = action_size
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.layer1 = nn.Linear(self.state_space, self.fc1_units, bias=True)
        self.layer2 = nn.Linear(self.fc1_units, self.fc2_units, bias=True)
        self.layer3 = nn.Linear(self.fc2_units, self.action_size, bias=True)

        # example layers
        # check own
        # nn.Conv2d(input_shape, 32, kernel_size=8, stride=4),
        # nn.ReLU(),
        # nn.Conv2d(32, 64, kernel_size=4, stride=2),
        # nn.ReLU(),
        # nn.Conv2d(64, 64, kernel_size=3, stride=1),
        # nn.ReLU()
        #
        # nn.Linear(conv_out_size, 512),
        # nn.ReLU(),
        # nn.Linear(512, n_actions)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # TODO: modify this to make convolutional layers
        # TODO: make for each stock

        layer1 = F.relu(self.layer1(state))
        layer2 = F.relu(self.layer2(layer1))
        layer3 = self.layer3(layer2)

        # TODO: undertsand logic:
        # should return an action under form: tensor[real no, real no] - so signal [real no, real no] for each stock
        # one signal is between [-1, 1]: strong sell or strong buy signal -> * hmax to tell how many shares to sell
        # decrease hmax for selling a smaller amount of shares
        return layer3

#TODO: remove unnecessary variables. Some come from Portfolio
class DQN_Agent(Portfolio):

    def __init__(self, stock_dim, action_dict, state_space, action_size, seed, buffer_size, batch_size, learning_rate, update_every, gamma, tau, balance, epsilon, epsilon_min, epsilon_decay, is_eval=False, model_name=""):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_space = state_space #no of unique tickers
        self.action_size = action_size # hold, buy, sell
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=100)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.balance = balance
        self.Q_network = QNetwork(self.state_size, self.action_size, seed )
        self.Q_network_val = QNetwork(self.state_size, self.action_size, seed)


        self.Q_network = MultiBranchNet(num_branches=stock_dim, r=stock_dim, c=28, output_size=action_dict)
        self.Q_network_val = MultiBranchNet(num_branches=stock_dim, r=stock_dim, c=28, output_size=action_dict)

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # self.model = torch.load('saved_models/{}.h5'.format(model_name)) if is_eval else QNetwork(self.state_space, self.action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=self.learning_rate)

        self.steps_until_update = 0


    # def reset(self):
    #     self.reset_portfolio()
    #     self.epsilon = 1.0 # reset exploration rate
    #
    # def remember(self, state, actions, reward, next_state, done):
    #     self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.Q_network(state)
        uniform_random = random.random()

        if (uniform_random > self.epsilon):
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = np.random.randint(self.action_size)
        return action

    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)
        self.steps_until_update = (self.steps_until_update + 1) % self.update_every

        if (self.steps_until_update == 0):
            if (self.memory.__len__() > self.batch_size):
                sample = self.memory.sample()
                states, actions, reward, next_state, done = sample

                self.Q_network_val.eval()
                with torch.no_grad():



                for i in range(len(outputs)):
                    q_value, max_index = torch.max(outputs[i], dim=1)

                    target_rewards = reward + self.gamma * q_value * (1 - done)

                    self.Q_network.train()
                    expected_rewards = self.Q_network.forward(states).gather(1, actions)
                    loss = F.mse_loss(expected_rewards, target_rewards)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    for Q_network_val_parameters, Q_network_parameters in zip(self.Q_network_val.parameters(),
                                                                              self.Q_network.parameters()):
                        Q_network_val_parameters.data.copy_(
                            self.tau * Q_network_parameters.data + (1.0 - self.tau) * Q_network_val_parameters.data)
        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class ReplayBuffer:
    """Fixed-size buffer to store past experience tuples, allowing the agent to learn from them."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.position = 0
        self.memory = []
        self.transition = namedtuple("Transition", field_names=["state", "action", "reward", "next_state", "done"])

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.current_transition = self.transition(state, action, reward, next_state, done)
        if (len(self.memory) < self.buffer_size):
            self.memory.append(None)
        self.memory[self.position] = self.current_transition
        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        sample = random.sample(self.memory, self.batch_size)
        sample_list = self.transition(*zip(*sample))
        states = torch.from_numpy(np.vstack(sample_list.state)).float()
        actions = torch.from_numpy(np.vstack(sample_list.action)).long()
        rewards = torch.from_numpy(np.vstack(sample_list.reward)).float()
        next_states = torch.from_numpy(np.vstack(sample_list.next_state)).float()
        dones = torch.from_numpy(np.vstack(sample_list.done).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

