from collections import deque

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import random
import datetime
from datetime import datetime
import pandas as pd

from src.config_model_DQN import INITIAL_AMOUNT

def getState(data_window, t, prev_closing_prices, agent):
    data_window = data_window.fillna(0.0)
    # prev_closing_prices = list of prev_closing_prices
    # encode the date as a number
    reference_date = datetime(2008,9,15)
    data_window['date'] = pd.to_datetime(data_window['date'])
    data_window['Date_Encoded'] = (data_window['date'] - reference_date).dt.days
    data_window = data_window.drop(['date'], axis=1)

    # one-hot-encode the 'TIC' aka stock name
    one_hot = pd.get_dummies(data_window['tic'])
    data_window = data_window.drop('tic', axis=1)
    # Join the encoded df
    data_window = data_window.join(one_hot)
    data_window_arr = data_window.to_numpy()

    if t == 0:

        portfolio_arr = agent.portfolio_state.T

        state = np.column_stack((data_window_arr,portfolio_arr))

    else: # we update options based on last closing prices
        # closing_prices = data_window['close'].tolist()
        #
        # for i in range(data_window.shape[0]): # iterate for each stock
        #     # no of shares for each stock: position/prev_closing_price * current closing price
        #     no_shares = agent.portfolio_state[1,i]/float(prev_closing_prices[i])
        #     agent.portfolio_state[1,i] = no_shares * closing_prices[i]

        portfolio_arr = agent.portfolio_state.T
        state = np.column_stack((data_window_arr, portfolio_arr)).astype(np.float64)

    return state, agent


class Portfolio:
    def __init__(self, num_stocks, balance, closing_prices=[]):
        self.initial_total_balance = float(balance)
        self.initial_total_balances = [float(balance)] * num_stocks # pos in stocks + cash left
        self.initial_position_stocks = [0.0]*num_stocks
        self.initial_position_portfolio = [0.0]*num_stocks
        self.initial_daily_return_stock = [0.0] * num_stocks
        self.initial_daily_return_total = [0.0] * num_stocks
        self.initial_cash_left = [float(balance)] * num_stocks
        self.initial_percentage_positions = [0.0] * num_stocks # liquid cash is included

        if not closing_prices:
            self.initial_shares_per_stock = [0.0] * num_stocks
        else:
            self.initial_shares_per_stock = (np.array(self.initial_position_portfolio)/np.array(closing_prices)).tolist()

        self.initial_portfolio_state = np.array([self.initial_total_balances,self.initial_position_stocks,
                                                 self.initial_position_portfolio,
                                                 self.initial_daily_return_stock, self.initial_daily_return_total,
                                                 self.initial_cash_left, self.initial_percentage_positions, self.initial_shares_per_stock])


        # subject to change while the agent explores
        self.total_balance = float(balance)
        self.total_balances = [float(balance)] * num_stocks  # pos in stocks + cash left
        self.position_stocks = [0.0]*num_stocks


        self.position_portfolio = [0.0]*num_stocks
        self.daily_return_stock = [0.0]*num_stocks
        self.daily_return_total = [0.0] * num_stocks
        self.percentage_positions = [0.0] * num_stocks # liquid cash is not included
        self.cash_left = [float(balance)] * num_stocks

        if not closing_prices:
            self.shares_per_stock = [0.0] * num_stocks
        else:
            self.shares_per_stock = (np.array(self.position_portfolio)/np.array(closing_prices)).tolist()

        self.initial_portfolio_state = np.array([self.initial_total_balances,self.initial_position_stocks,
                                                 self.initial_position_portfolio,
                                                 self.initial_daily_return_stock, self.initial_daily_return_total,
                                                 self.initial_cash_left, self.initial_percentage_positions, self.initial_shares_per_stock])

        self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
                                             self.daily_return_stock,self.daily_return_total,
                                             self.cash_left, self.percentage_positions, self.shares_per_stock])

    def reset_portfolio(self):
        self.portfolio_state = np.copy(self.initial_portfolio_state)


class DQNNetwork(nn.Module):
    def __init__(self, num_stocks, num_actions, num_features):
        super().__init__()

        self.num_stocks = num_stocks
        self.num_features = num_features
        self.h_number = int(np.floor(2/3*self.num_features)) #recommended size
        self.f_number = self.h_number * 2 #since input will be double
        self.num_actions = num_actions

        # define layers
        self.linear_h = nn.Linear(self.num_features, self.h_number)
        self.linear_f = nn.Linear(self.f_number, self.num_actions)
        self.activation = torch.nn.ReLU()

    def forward(self,x):
        h_outputs = []
        f_outputs = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if type(x)== tuple:
            x = torch.from_numpy(x[0]).to(device)
        else:
            x = torch.from_numpy(x).to(device)
        for i in range(self.num_stocks):
            x_i = x[i,:].float()
            x_i = self.linear_h(x_i)
            x_i = self.activation(x_i)
            h_outputs.append(x_i)
        for i in range(self.num_stocks):
            h_outputs_stock_i = h_outputs[i]
            h_outputs_temp = h_outputs.copy()
            h_outputs_temp.pop(i)
            h_outputs_without_i_tensor = torch.vstack(h_outputs_temp)
            # create a vector with h_outputs for stock i and mean of stocks i+1,i+2...in
            f_input_i = [h_outputs_stock_i, torch.mean(h_outputs_without_i_tensor, 0, True)]
            f_input_i = torch.vstack(f_input_i)
            #print(f_input_i)
            f_input_i = torch.flatten(f_input_i)
            # pass through network of size f_number
            f_input_i = self.linear_f(f_input_i)  #row of num_actions for each stock entry in the Q table
            f_input_i = self.activation(f_input_i)
            f_outputs.append(f_input_i)

        x = torch.vstack(f_outputs)
        #  final flattened output of size (num_stocks x num_actions)
        x = torch.flatten(x)

        return x

class Agent(Portfolio):
    def __init__(self, num_stocks, actions_dict, num_features, balance, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.9999, learning_rate=0.001, batch_size=32, tau=1e-3, num_epochs=10):
        super().__init__(balance=balance, num_stocks=num_stocks)

        self.num_stocks = num_stocks
        self.num_actions = len(actions_dict)
        self.actions_dict = actions_dict
        self.actions_dict = actions_dict
        self.gamma = gamma
        self.tau = tau # Q network target update
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = []

        #self.buffer_size = 60
        self.batch_size = batch_size
        self.batch_loss_history = []
        self.epoch_numbers = []
        self.num_features_from_data = num_features
        self.num_features_total = self.num_features_from_data + self.portfolio_state.shape[0]
        self.num_epochs = num_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q_network = DQNNetwork(num_stocks, self.num_actions, self.num_features_total).to(self.device).float()
        self.Q_network_val = DQNNetwork(num_stocks, self.num_actions, self.num_features_total).to(self.device).float()

        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), lr=self.learning_rate)

        self.loss_fn = torch.nn.MSELoss()

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def reset(self):
        self.reset_portfolio()
        print(f'Portfolio after reset is \n{self.portfolio_state}')

        self.epsilon = 1.0 # reset exploration rate

    def act(self, state):
        if not self.Q_network.training and np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions*self.num_stocks)
            logging.info(f" exploration stage.")

        options = self.Q_network(state)

        action_index = torch.argmax(options).item()

        return action_index

    def expReplay(self, epoch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # retrieve recent buffer_size long memory
        mini_batch = [self.memory[i] for i in range(len(self.memory) - self.batch_size + 1, len(self.memory))]
        running_loss = 0.0
        loss_history = []
        i = 0

        for state, actions, reward, next_state, done in mini_batch:

            if not done:
                self.Q_network_val.eval()
                with torch.no_grad():
                    value = torch.max(self.Q_network_val.forward(next_state)).item()
                    target_rewards = reward + self.gamma * torch.max(self.Q_network_val.forward(next_state)).item()
            else:
                target_rewards = reward

            self.Q_network.train()

            actions_tensor = torch.tensor([actions[0]], dtype=torch.long).to(device)  # Convert actions[0] to a tensor
            expected_rewards = torch.gather(self.Q_network.forward(state), dim=0, index=actions_tensor).to(device)

            target_rewards = torch.tensor([target_rewards], dtype=torch.float).to(device)


            # CHANGE TO MAKE VECTOR?
            loss = F.mse_loss(expected_rewards, target_rewards)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Printing loss

        avg_loss = running_loss/self.batch_size
        self.batch_loss_history.append(avg_loss)

        # Save the model parameters after training
        if epoch == self.num_epochs - 1:
            current_date = datetime.now()

            # Format the date as a readable string
            date_string = current_date.strftime("%d_%m_%Y")
            torch.save(self.Q_network.state_dict(), f'./models/trained_agent_{date_string}.pt')

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # update at the end of batch Q_network_val using tau
        for Q_network_val_parameters, Q_network_parameters in zip(self.Q_network_val.parameters(),
                                                                  self.Q_network.parameters()):
            Q_network_val_parameters.data.copy_(
                self.tau * Q_network_parameters.data + (1.0 - self.tau) * Q_network_val_parameters.data)

    def execute_action(self, action_index_for_stock_i, closing_prices, stock_i, h):
        action_dictionary = {
            0: self.buy_0_1,
            1: self.buy_0_25,
            2: self.buy_0_50,
            3: self.buy_0_75,
            4: self.buy_1,
            5: self.sell_0_1,
            6: self.sell_0_25,
            7: self.sell_0_50,
            8: self.sell_0_75,
            9: self.sell_1,
            10: self.hold,
            11: self.sell_everything,
            12: self.buy_1_share
        }
        selected_function = action_dictionary.get(action_index_for_stock_i)
        if selected_function is not None:
            # Call the selected function with the provided arguments
            reward = selected_function(closing_prices, stock_i, h)
            # Process the result if needed
        else:
            # Handle the case when action_index_for_stock_i is not found in the dictionary
            reward = "Invalid action index"

        return reward

    def buy_0_1(self, closing_prices, stock_i, h):
        '''
        Function to buy 0.1*h_cash of stock_i

        '''

        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8, num_stocks)

        #indexes
        # total_balance -> self.portfolio_state[0,:]
        # (cash) position per stock -> self.portfolio_state[1,:]
        # (cash) position portfolio -> self.portfolio_state[2,:]
        # daily return per stock -> self.portfolio_state[3,:]
        # daily return total portfolio -> self.portfolio_state[4,:]
        # cash left total -> self.portfolio_state[5,:]
        # percentage position per stock -> self.portfolio_state[6,:]
        # shares per stock -> self.portfolio_state[7,:]

        # TODO: revise logic
        # store prev stock position
        prev_position_stocks = self.portfolio_state[1,:]

        # store prev portfolio position # we can take the first element since it is the same value for the whole row
        prev_position_portfolio = self.portfolio_state[2,0]

        # UPDATE DAILY RETURNS ##############################
        # update all stock positions based on new closing prices: no.shares * closing prices
        for i in range(self.portfolio_state.shape[1]):
            self.portfolio_state[1, i] = self.portfolio_state[7,i]*closing_prices[i]l

        # update position total (same for all): sum of all stock positions
        for i in range(self.portfolio_state.shape[1]):
            self.portfolio_state[2, i] = np.sum(self.portfolio_state[1, :])

        # update total balance: position total + cash left
        for i in range(self.portfolio_state.shape[1]):
            self.portfolio_state[0,i] = self.portfolio_state[2,0] + self.portfolio_state[4,0]

        # update daily return per stocks: position stock_i - prev position stock i
        for i in range(self.portfolio_state.shape[1]):
            self.portfolio_state[3,i] = self.portfolio_state[1,stock_i] - prev_position_stocks[i]

        # update daily return total portfolio (same for all): position_portfolio - prev position portfolio
        for i in range(self.portfolio_state.shape[1]):
            self.portfolio_state[4,i] = self.portfolio_state[2,0] - prev_position_portfolio

        # BUYING AND SELLING IS DONE AFTER OBSERVING DAILY RETURNS BASED ON PRICE CHANGE ################

        # change the value of position stock_i based on extra amount which is bought
        buy_amount_stock_i = 0.1 * h
        self.portfolio_state[1,stock_i] = self.portfolio_state[1,stock_i] + buy_amount_stock_i

        # update position total (same for all): sum of all stock positions
        for i in range(self.portfolio_state.shape[1]):
            self.portfolio_state[2, i] = np.sum(self.portfolio_state[1, :])

        # update cash left (same for all)
        for i in range(self.portfolio_state.shape[1]):
            self.portfolio_state[5, i] = self.initial_total_balance - buy_amount_stock_i

        # update percentage of stock position: position stock/total balance
        self.portfolio_state[6, :] = self.portfolio_state[1,:]/self.portfolio_state[0,:]

        # update total balance: position total + cash left
        for i in range(self.portfolio_state.shape[1]):
            self.portfolio_state[0,i] = self.portfolio_state[2,0] + self.portfolio_state[4,0]

        # update new stock shares: stock positions/ closing prices
        for i in range(self.portfolio_state.shape[1]):
            self.portfolio_state[7, i] = self.portfolio_state[1,i]/closing_prices[i]

        # reward is daily/ min return per protfolio
        reward = self.portfolio_state[4,0]

        return reward



    def buy_0_25(self, closing_prices, stock_i, h):


    def buy_0_50(self, closing_prices, stock_i, h):


    def buy_0_75(self, closing_prices, stock_i, h):


    def buy_1(self, closing_prices, stock_i, h):


    def sell_0_1(self, close_price_stock_i, stock_i, h):



    def sell_0_25(self, close_price_stock_i, stock_i, h):


    def sell_0_50(self, close_price_stock_i, stock_i, h):


    def sell_0_75(self, close_price_stock_i, stock_i, h):


    def sell_1(self, close_price_stock_i, stock_i, h):


    def hold(self, close_price_stock_i, stock_i, h):


    def sell_everything(self, closing_prices, stock_i, h):
        # self.portfolio_state = np.array([self.total_balance,self.position_stocks,self.position_portfolio,
        #                                  self.daily_return_stock,self.daily_return_total,
        #                                  self.cash_left, self.percentage_positions])
        # shape: (7, num_stocks)

        # indexes
        # total_balance -> self.portfolio_state[0,:]
        # (cash) position per stock -> self.portfolio_state[1,:]
        # (cash) position portfolio -> self.portfolio_state[2,:]
        # daily return per stock -> self.portfolio_state[3,:]
        # daily return total portfolio -> self.portfolio_state[4,:]
        # cash left total -> self.portfolio_state[5,:]
        # percentage position per stock -> self.portfolio_state[6,:]

        #  total balance after selling
        balance_sold = self.portfolio_state[0, 0]

        self.reset_portfolio()

        # update total balance: position plus cash left
        for i in range(self.portfolio_state.shape[1]):
            self.portfolio_state[0, i] = balance_sold

        # update cash left balance:
        for i in range(self.portfolio_state.shape[1]):
            self.portfolio_state[5, i] = balance_sold

        #TODO: revise?
        reward = 0

        return reward

    def buy_1_share(self, close_price_stock_i, stock_i, h):



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
    #     q_values = self.Q_network(state)
    #     q_value = q_values.gather(1, action).squeeze(1)
    #     loss = self.loss_fn(q_value, target_Q)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.Q_table = target_Q.detach().cpu().numpy()
    #
    # def decay_epsilon(self):
    #     self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

