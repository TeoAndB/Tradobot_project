from collections import deque

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import random
import datetime

from src.config_model_DQN import initial_amount


class Portfolio:
    def __init__(self, num_stocks, balance=initial_amount):
        self.initial_portfolio_value = balance
        self.balance = balance
        self.num_stocks = num_stocks
        self.shares = [0]*num_stocks
        self.shares_list = []
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



    #TODO: write all functions and move them to a different python file which you import

def buy_0_1(agent, stock_i, h):

def buy_0_25(agent, stock_i, h):

def buy_0_50(agent, stock_i, h):

def buy_0_75(agent, stock_i, h):


def buy_1(agent, stock_i, h):


def sell_0_1(agent, stock_i, h):


def sell_0_25(agent, stock_i, h):


def sell_0_50(agent, stock_i, h):


def sell_0_75(agent, stock_i, h):


def sell_1(agent, stock_i, h):

def hold(agent, stock_i, h):

def default():
    return "Invalid transaction"

class DQNNetwork(nn.Module):
    def __init__(self, num_stocks, num_actions, num_features):
        super(DQNNetwork, self).__ionit__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.h_number = int(np.floor(2/3*self.num_stocks * self.num_features)) #recommended size
        self.f_number = self.h_number * 2 #since input will be double
        self.num_actions = num_actions

        # define layers
        self.linear_h = nn.Linear(self.num_features, self.h_number)
        self.linear_f = nn.Linear(self.f_number, self.num_actions)

        def forward(self,x):
            h_outputs = nn.ModuleList()
            f_outputs = nn.ModuleList()
            for i in range(num_stocks):
                x_i = x[i][:]
                x_i = torch.linear_h(x_i)
                x_i = torch.relu(x_i)
                h_outputs.append(x_i)
            for i in range(num_stocks):
                h_outputs_without_i = h_outputs[:i] + h_outputs[(i+1):] #list of tensors without i
                h_outputs_without_i_tensor = torch.cat(h_outputs_without_i, dim=1)
                f_input_i = [h_outputs[i], torch.mean(h_outputs_without_i_tensor)]
                f_input_i = torch.cat(f_input_i, dim=1)
                f_input_i = torch.flatten(f_input_i)
                # pass through network of size f_number
                f_input_i = self.linear_f(f_input_i)  #row of num_actions for each stock entry in the Q table
                f_outputs.append(f_input_i)

            x = torch.cat(f_outputs, dim=1)
            x = torch.t(x)
            #  final output of size (num_stocks, num_actions)
            return x

        import torch
        import numpy as np

class Agent(Portfolio):
    def __init__(self, num_stocks, num_actions, num_features, balance, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.9999, learning_rate=0.001, batch_size=32):
        super().__init__(balance=balance, num_stocks=num_stocks)
        self.num_stocks = num_stocks
        self.num_actions = num_actions
        self.num_features = num_features
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.loss_history = []
        self.memory = []

        #self.buffer_size = 60
        self.batch_size = batch_size
        self.batch_loss_history = []
        self.epoch_numbers = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(num_stocks, num_actions, num_features).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.loss_fn = torch.nn.MSELoss()

        # Initialize Q-table to zeros
        self.Q_table = np.zeros((num_stocks, num_actions))

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0 # reset exploration rate

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        options = self.model(state)

        # returns (action_stock1, action_stock2, action_stock3)
        return np.argmax(options[0])

    def expReplay(self, epoch):
        # retrieve recent buffer_size long memory
        mini_batch = [self.memory[i] for i in range(len(self.memory) - self.batch_size + 1, len(self.memory))]
        running_loss = 0.0
        loss_history = []
        i = 0

        for state, action, reward, next_state, done in mini_batch:
            q_target = reward # final state
            if not done:
                with torch.no_grad():
                    # learns from one step in the future
                    q_target = reward + self.gamma * np.amax(self.model(next_state)[0]) # (q_val1', q_val_2', q_val3')
            self.model.train()

            # predicted Q value for the curr state at the given action
            q_pred = self.Q_network.forward(state).gather(1, action) # (q_val1, q_val2 , q_val3)

            loss = self.loss_fn(q_pred, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                avg_loss = running_loss/100
                self.batch_loss_history.append(avg_loss)
                self.epoch_numbers.append(epoch+1)
                running_loss = 0.0

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f}")

            # Save the model parameters after training
            if epoch == self.num_epochs - 1:
                current_date = datetime.now()

                # Format the date as a readable string
                date_string = current_date.strftime("%d_%m_%Y")
                torch.save(self.model.state_dict(), f'./models/trained_agent_{date_string}.pt')

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

                # for Q_network_val_parameters, Q_network_parameters in zip(self.Q_network_val.parameters(), self.Q_network.parameters()):
            #     Q_network_val_parameters.data.copy_(self.tau * Q_network_parameters.data + (1.0 - self.tau) * Q_network_val_parameters.data)

    # uses defined transaction actions based on action index
    def execute_transaction(self, action, stock_i, h):
        switch_dict = {
            0: buy_0_1,
            1: buy_0_25,
            2: buy_0_50,
            3: buy_0_75,
            4: buy_1,
            5: sell_0_1,
            6: sell_0_25,
            7: sell_0_50,
            8: sell_0_75,
            9: sell_1,
            10: hold
        }
        func = switch_dict.get(action, default)
        return func(self, stock_i, h)

#############################################################333333
    # def act(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         return np.random.choice(self.num_actions, self.num_stocks)
    #     q_values = self.Q_table[state, :]
    #     return np.argmax(q_values)
    #
    #
    #
    # def experience_replay(self, state, action, reward, next_state, done):
    #     target_Q = self.Q_table.copy()
    #     next_action = self.act(next_state)
    #     target_Q[state, action] = reward + (1 - done) * self.gamma * self.Q_table[next_state, next_action]
    #     target_Q = torch.tensor(target_Q, dtype=torch.float32).to(self.device)
    #     state = torch.tensor(state, dtype=torch.float32).to(self.device)
    #     action = torch.tensor(action, dtype=torch.int64).unsqueeze(0).to(self.device)
    #     q_values = self.model(state)
    #     q_value = q_values.gather(1, action).squeeze(1)
    #     loss = self.loss_fn(q_value, target_Q)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.Q_table = target_Q.detach().cpu().numpy()
    #
    # def decay_epsilon(self):
    #     self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

