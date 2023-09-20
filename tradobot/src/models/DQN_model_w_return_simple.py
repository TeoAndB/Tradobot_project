from collections import deque

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import random
import datetime
from datetime import datetime
import pandas as pd
import random
import logging
from collections import Counter


from src.config_model_DQN_return import INITIAL_AMOUNT, NUM_ACTIONS, TIME_LAG, NUM_STOCKS, WEIGHT_DECAY, NUM_SAMPLING

total_balance_idx = 0
position_stock_idx = 1
position_portfolio_idx = 2
return_stock_idx = 3
return_portfolio_idx = 4
cash_left_idx = 5
percentage_position_stock_idx = 6
shares_stock_idx = 7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getMeanStdFromData(df, NUM_STOCKS):
    scalar_columns = df.select_dtypes(exclude='object').columns

    # Calculate mean every 3rd row for scalar columns
    means = []
    std_devs = []
    for i in range(NUM_STOCKS):
        means.append(df.iloc[i::3][scalar_columns].mean())
        std_devs.append(df.iloc[i::3][scalar_columns].std())

    mu_dataframe = pd.DataFrame(means)
    std_dataframe = pd.DataFrame(std_devs)

    return mu_dataframe, std_dataframe

# device = torch.device('cpu')

def getState(data_window, t, agent, mu_df, std_df):
    data_window = data_window.fillna(0.0)
    # prev_closing_prices = list of prev_closing_prices
    # encode the date as a number signifying day interval length between curr date and past date

    data_window = data_window.drop(['date','day'], axis=1)

    # one-hot-encode the 'TIC' aka stock name
    one_hot = pd.get_dummies(data_window['tic'])
    data_window = data_window.drop('tic', axis=1)
    # Join the encoded df
    data_window = data_window.join(one_hot)

    # retrieving the time series for the closing price
    data_window.rename(columns={'close': 'close_lag_0'}, inplace=True)
    close_price_column_names = [col for col in data_window.columns if col.startswith('close_lag')]
    close_price_column_names = close_price_column_names[::-1]
    close_price_columns_df = data_window.loc[:, close_price_column_names]
    close_price_columns_arr = close_price_columns_df.to_numpy()

    # excluding the lags from the original data_window
    filtered_columns = [col for col in data_window.columns if col not in close_price_column_names]
    filtered_columns.append('close_lag_0')
    data_window_no_lags = data_window.loc[:, filtered_columns]
    data_window_no_lags.rename(columns={'close_lag_0': 'close'}, inplace=True)

    # substract the mean and divide by standard deviation for scalar values
    common_columns = list(set(data_window_no_lags.columns) & set(mu_df.columns))
    for idx, row in data_window_no_lags.iterrows():
        data_window_no_lags.loc[idx, common_columns] = (row[common_columns] - mu_df.iloc[idx % NUM_STOCKS][common_columns]) / \
                                               std_df.iloc[idx % NUM_STOCKS][common_columns]

    data_window_arr = data_window_no_lags.to_numpy()

    if t == 0:

        portfolio_arr = np.array(agent.portfolio_state.copy()).T

        state = np.column_stack((data_window_arr, close_price_columns_arr))

        # state = np.column_stack((data_window_arr, portfolio_arr, close_price_columns_arr))

    else:  # we update options based on last closing prices

        portfolio_arr = np.array(agent.portfolio_state.copy()).T

        state = np.column_stack((data_window_arr, close_price_columns_arr)).astype(np.float)

        # state = np.column_stack((data_window_arr, portfolio_arr, close_price_columns_arr)).astype(np.float)

    return state


class Portfolio:
    def __init__(self, num_stocks, balance, name_stocks, closing_prices=[]):

        self.timestamp_portfolio = 'None'
        self.name_stocks_list = name_stocks
        self.initial_total_balance = float(balance)
        self.initial_total_balances = [float(balance)] * num_stocks  # pos in stocks + cash left
        self.initial_position_stocks = [0.0] * num_stocks
        self.initial_position_portfolio = [0.0] * num_stocks
        self.initial_daily_return_stock = [0.0] * num_stocks
        self.initial_daily_return_total = [0.0] * num_stocks
        self.initial_cash_left = [float(balance)] * num_stocks
        self.initial_percentage_positions = [0.0] * num_stocks  # liquid cash is included

        # for explainability
        self.portfolio_state_rows = ['total_balance', 'position_per_stock', 'position_portfolio',
                                     'daily_return_per_stock', 'daily_return_portfolio', 'cash_left',
                                     'percentage_position_stock', 'shares_per_stock']

        self.explainability_df = pd.DataFrame()

        if not closing_prices:
            self.initial_shares_per_stock = [0.0] * num_stocks
        else:
            self.initial_shares_per_stock = (
                    np.array(self.initial_position_portfolio) / np.array(closing_prices)).tolist()

        self.initial_portfolio_state = np.array([self.initial_total_balances, self.initial_position_stocks,
                                                 self.initial_position_portfolio,
                                                 self.initial_daily_return_stock, self.initial_daily_return_total,
                                                 self.initial_cash_left, self.initial_percentage_positions,
                                                 self.initial_shares_per_stock])

        # subject to change while the agent explores
        self.total_balance = float(balance)
        self.total_balances = [float(balance)] * num_stocks  # pos in stocks + cash left
        self.position_stocks = [0.0] * num_stocks

        self.position_portfolio = [0.0] * num_stocks
        self.daily_return_stock = [0.0] * num_stocks
        self.daily_return_total = [0.0] * num_stocks
        self.percentage_positions = [0.0] * num_stocks  # liquid cash is not included
        self.cash_left = [float(balance)] * num_stocks

        if not closing_prices:
            self.shares_per_stock = [0.0] * num_stocks
        else:
            self.shares_per_stock = (np.array(self.position_portfolio) / np.array(closing_prices)).tolist()

        self.initial_portfolio_state = np.array([self.initial_total_balances, self.initial_position_stocks,
                                                 self.initial_position_portfolio,
                                                 self.initial_daily_return_stock, self.initial_daily_return_total,
                                                 self.initial_cash_left, self.initial_percentage_positions,
                                                 self.initial_shares_per_stock])

        self.portfolio_state = np.asarray([self.total_balances, self.position_stocks, self.position_portfolio,
                                           self.daily_return_stock, self.daily_return_total,
                                           self.cash_left, self.percentage_positions, self.shares_per_stock])

    def reset_portfolio(self):
        self.portfolio_state = np.copy(self.initial_portfolio_state)



class DQNNetwork(nn.Module):
    def __init__(self, num_stocks, num_actions, num_features, batch_size):
        super().__init__()

        self.num_stocks = num_stocks
        self.num_features = num_features
        self.num_actions = num_actions
        self.num_actions_all = num_actions * num_stocks

        # Calculate flattened input size
        flattened_size = self.num_stocks * (self.num_features - TIME_LAG)

        # Calculate intermediate size (you can adjust this value as needed)
        hidden_size = int(flattened_size / 2)  # Just a heuristic; feel free to change

        # define layers
        self.linear_h = nn.Linear(flattened_size, hidden_size)
        self.linear_f = nn.Linear(hidden_size, self.num_actions_all)  # output dimension matches self.num_actions_all
        self.activation = torch.nn.ReLU()

        # Initializing the weights with the Xavier initialization method
        torch.nn.init.normal_(self.linear_h.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.linear_h.weight, mean=0, std=0.01)

    def forward(self, x):
        # Ensure the input is a torch Tensor and on the correct device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device)
        else:
            x = x.to(device)

        # Flatten the input tensor
        x = x[:, :, :-(TIME_LAG)].float().reshape(x.shape[0],
                                                  -1)  # Resultant shape: [batch_size, num_stocks*(num_features-TIME_LAG)]

        # Pass through the first network
        x = self.linear_h(x)


        x = self.activation(x)

        # Pass through the second network
        x = self.linear_f(x)
        # x = self.activation_2(x)

        return x


class Agent(Portfolio):
    def __init__(self, num_stocks, actions_dict, h, num_features, balance, name_stocks, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.9999, learning_rate=0.001, batch_size=32, tau=1e-3, num_epochs=10, model_path='',
                 model_target_path=''):
        super().__init__(balance=balance, num_stocks=num_stocks, name_stocks=name_stocks)

        self.num_stocks = num_stocks
        self.utility = 0.0
        self.num_actions = len(actions_dict)
        self.actions_dict = actions_dict
        self.actions_dict = actions_dict
        self.h = h
        self.reward = 0.0
        self.gamma = gamma
        self.tau = tau  # Q network target update
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = []

        # self.buffer_size = 60
        self.batch_size = batch_size
        self.batch_loss_history = []
        self.epoch_numbers = []
        self.num_features_from_data = num_features
        # self.num_features_total = self.num_features_from_data + self.portfolio_state.shape[0]
        self.num_features_total = self.num_features_from_data

        self.num_epochs = num_epochs

        self.device = device

        self.Q_network = DQNNetwork(num_stocks, self.num_actions, self.num_features_total, self.batch_size).to(
            self.device).float()
        self.Q_network_val = DQNNetwork(num_stocks, self.num_actions, self.num_features_total, self.batch_size).to(
            self.device).float()

        # load a pre-trained model
        if model_path and model_target_path:
            self.Q_network.load_state_dict(torch.load(model_path))
            self.Q_network_val.load_state_dict(torch.load(model_target_path))

        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), lr=self.learning_rate)

        self.loss_fn = torch.nn.MSELoss()
        # self.torch.nn.HuberLoss()



    def remember(self, state, actions, closing_prices, reward, next_state, done):
        self.memory.append((state, actions, closing_prices, reward, next_state, done))

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0  # reset exploration rate
        self.memory = []
        self.batch_loss_history = []
        self.utility = 0.0

    def soft_reset(self):
        # self.epsilon = 1.0  # reset exploration rate
        self.reset_portfolio()
        self.memory = []
        self.batch_loss_history = []
        self.utility = 0.0

    def act_deterministic(self, state, closing_prices):
        # state = state.reshape((-1,self.num_stocks * self.num_features_total))
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.numpy()

        with torch.no_grad():
            action_values = self.Q_network(state)

        options_allowed = self.maskActions(action_values, self.h, closing_prices)

        action_index = torch.argmax(options_allowed).item()

        return action_index

    def act(self, state, closing_prices):
        # state = state.reshape((-1,self.num_stocks * self.num_features_total))
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.numpy()

        with torch.no_grad():
            action_values = self.Q_network(state)
        uniform_random = np.random.rand()

        if (uniform_random > self.epsilon):

            options_allowed = self.maskActions(action_values, self.h, closing_prices, train=True)

            action_index = torch.argmax(options_allowed).item()

        else:

            random_options = torch.randint(low=0, high=self.num_stocks * self.num_actions,
                                           size=(self.num_stocks, self.num_actions))
            random_options = torch.flatten(random_options)
            random_options_allowed = self.maskActions(random_options, self.h, closing_prices)

            action_index = torch.argmax(random_options_allowed).item()

        return action_index

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        sample = random.sample(self.memory, self.batch_size)
        statess, actionss, closing_pricess, rewardss, next_statess, doness = zip(*sample)

        states = np.stack(statess, axis=0)
        actions = np.stack(actionss, axis=0)
        closing_prices = np.stack(closing_pricess, axis=0)
        rewards = np.stack(rewardss, axis=0)
        next_states = np.stack(next_statess, axis=0)
        dones = np.stack(doness, axis=0)

        # self.Q_network_val.forward(ss).shape
        #
        # self.Q_network.forward(state[0])
        return (states, actions, rewards, next_states, dones)


    def CQL_loss(self, expected_rewards, target_rewards, actions_indexes, states):
        # log term of the CQL
        q_outputs = self.Q_network.forward(states)
        sum_exp_outputs = torch.log(torch.sum(torch.exp(q_outputs)))


        # q value according to action distribution of sample
        counter = Counter(actions_indexes.cpu().numpy())  # Convert tensor to CPU and then to numpy
        predominant_action_idx = counter.most_common(1)[0][0]
        actions_indexes_cpu = actions_indexes.cpu()
        actions_indexes_predominant = np.full(actions_indexes_cpu.shape[0], predominant_action_idx)
        actions_indexes_predominant = torch.from_numpy(actions_indexes_predominant).to(device)
        actions_indexes_predominant = torch.unsqueeze(actions_indexes_predominant, dim=1)

        q_values_predominant = self.Q_network.forward(states).gather(1, actions_indexes_predominant)
        # expectation:
        expectation_q_values_predominant = torch.sum(q_values_predominant)

        loss_term = 0.5 * torch.mean((expected_rewards - target_rewards) ** 2)
        return sum_exp_outputs - expectation_q_values_predominant + loss_term

    def expReplay(self, epoch):

        # retrieve recent buffer_size long memory
        sample = self.sample()
        states, actions, rewards, next_states, dones = sample
        self.Q_network_val.eval()
        with torch.no_grad():
            target_rewards = (rewards + self.gamma * (
                torch.max(self.Q_network_val.forward(next_states), dim=1, keepdim=True)[0]).cpu().numpy() * (
                                          1 - dones))[0]

        self.Q_network.train()
        actions_indexes = actions[:, 0]
        actions_indexes = torch.from_numpy(actions_indexes).to(device)
        action_indexes = torch.unsqueeze(actions_indexes, dim=1)
        expected_rewards = self.Q_network.forward(states).gather(1, action_indexes)
        target_rewards = torch.tensor(target_rewards, device=device).float().view(-1).unsqueeze(1)
        # target_rewards = torch.tensor(target_rewards, device=device).float().view(-1).unsqueeze(1)


        loss = self.loss_fn(expected_rewards, target_rewards)
        # loss = self.CQL_loss(expected_rewards, target_rewards, actions_indexes, states)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        avg_loss = loss.mean().detach().cpu().numpy()
        self.batch_loss_history.append(avg_loss)

        for Q_network_val_parameters, Q_network_parameters in zip(self.Q_network_val.parameters(),
                                                                  self.Q_network.parameters()):
            Q_network_val_parameters.data.copy_(
                self.tau * Q_network_parameters.data + (1.0 - self.tau) * Q_network_val_parameters.data)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def expReplay_validation(self, epoch):

        # retrieve recent buffer_size long memory
        sample = self.sample()
        states, actions, rewards, next_states, dones = sample
        self.Q_network_val.eval()

        with torch.no_grad():
            target_rewards = (rewards + self.gamma * (
                torch.max(self.Q_network_val.forward(next_states), dim=1, keepdim=True)[0]).cpu().numpy() * (
                                          1 - dones))[0]

        # self.Q_network.train()
        actions_indexes = actions[:, 0]
        actions_indexes = torch.from_numpy(actions_indexes).to(device)
        action_indexes = torch.unsqueeze(actions_indexes, dim=1)
        expected_rewards = self.Q_network.forward(states).gather(1, action_indexes)
        target_rewards = torch.tensor(target_rewards, device=device).float().view(-1).unsqueeze(1)

        loss = self.loss_fn(expected_rewards, target_rewards)
        # loss = self.CQL_loss(expected_rewards, target_rewards, actions_indexes, states)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        avg_loss = loss.mean().detach().cpu().numpy()
        self.batch_loss_history.append(avg_loss)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def execute_action(self, action_index_for_stock_i, closing_prices, stock_i, h, e, dates):
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
            amount_transaction = selected_function(closing_prices, stock_i, h, e, dates)
            # Process the result if needed
        else:
            # Handle the case when action_index_for_stock_i is not found in the dictionary
            print("Invalid action index")

        return amount_transaction


    def update_portfolio(self, next_closing_prices, next_dates, stock_i, amount_transaction):

        # Explanation:
        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8, num_stocks)

        # indexes
        # total_balance -> self.portfolio_state[total_balance_idx,:]
        # (cash) position per stock -> self.portfolio_state[position_stock_idx,:]
        # (cash) position portfolio -> self.portfolio_state[position_portfolio_idx,:]
        # daily return per stock -> self.portfolio_state[return_stock_idx,:]
        # daily return total portfolio -> self.portfolio_state[return_portfolio_idx,:]
        # cash left total -> self.portfolio_state[cash_left_idx,:]
        # percentage position per stock -> self.portfolio_state[percentage_position_stock_idx,:]
        # shares per stock -> self.portfolio_state[shares_stock_idx,:]

        self.timestamp_portfolio = next_dates[0]
        prev_balance = self.portfolio_state[total_balance_idx, 0].copy()

        # store prev stock position
        prev_position_stocks = self.portfolio_state[position_stock_idx, :].copy()

        # store prev portfolio position # we can take the first element since it is the same value for the whole row
        prev_position_portfolio = self.portfolio_state[position_portfolio_idx, 0].copy()

        # UPDATE DAILY RETURNS ##############################
        # update all stock positions based on new closing prices: no.shares * closing prices
        for i in range(NUM_STOCKS):
            self.portfolio_state[position_stock_idx, i] = self.portfolio_state[shares_stock_idx, i] * \
                                                          next_closing_prices[i]

        # update position total (same for all): sum of all stock positions
        self.portfolio_state[position_portfolio_idx, :] = np.sum(self.portfolio_state[position_stock_idx, :], axis=0)

        # update total balance: position total + cash left
        for i in range(NUM_STOCKS):
            self.portfolio_state[total_balance_idx, i] = self.portfolio_state[position_portfolio_idx, 0] + \
                                                         self.portfolio_state[cash_left_idx, 0]

        # update daily return per stocks: position stock - prev position stock
        for i in range(NUM_STOCKS):
            self.portfolio_state[return_stock_idx, i] = self.portfolio_state[position_stock_idx, i] - \
                                                        prev_position_stocks[i]

        # update daily return total portfolio (same for all): position_portfolio - prev position portfolio
        self.portfolio_state[return_portfolio_idx, :] = np.sum(self.portfolio_state[return_stock_idx, :])

        # reward is return per total balance

        if self.portfolio_state[position_portfolio_idx, 0] == prev_position_portfolio:
            reward = 0.0
        else:
            if amount_transaction:
                reward = (self.portfolio_state[total_balance_idx, 0].copy() - prev_balance)/prev_balance

                self.utility += reward

                return reward
            reward = 0.0

        self.utility += reward
        return reward


    def buy_0_1(self, closing_prices, stock_i, h, e, dates):

        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8,NUM_STOCKS)

        prev_balance = self.portfolio_state[total_balance_idx, 0]

        # change the value of position stock_i based on extra amount which is bought
        buy_amount_stock_i = 0.1 * h
        self.portfolio_state[position_stock_idx, stock_i] = self.portfolio_state[
                                                                position_stock_idx, stock_i] + buy_amount_stock_i

        # update position total (same for all): sum of all stock positions
        total_position = np.sum(self.portfolio_state[position_stock_idx, :])
        self.portfolio_state[position_portfolio_idx, :] = total_position

        # update cash left (same for all)
        self.portfolio_state[cash_left_idx, :] -= buy_amount_stock_i

        # update percentage of stock position: position stock/total balance
        self.portfolio_state[percentage_position_stock_idx, :] = self.portfolio_state[position_stock_idx,
                                                                 :] / self.portfolio_state[total_balance_idx, :]

        # update total balance: position portfolio + cash left
        self.portfolio_state[total_balance_idx, :] = total_position + self.portfolio_state[cash_left_idx, :]

        # update new stock shares: stock positions/ closing prices
        for i in range(NUM_STOCKS):
            self.portfolio_state[shares_stock_idx, i] = self.portfolio_state[position_stock_idx, i] / closing_prices[i]

        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'buy_0_1'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return buy_amount_stock_i

    def buy_0_25(self, closing_prices, stock_i, h, e, dates):

        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8,NUM_STOCKS)

        prev_balance = self.portfolio_state[total_balance_idx, 0]

        # change the value of position stock_i based on extra amount which is bought
        buy_amount_stock_i = 0.25 * h
        self.portfolio_state[position_stock_idx, stock_i] = self.portfolio_state[
                                                                position_stock_idx, stock_i] + buy_amount_stock_i

        # update position total (same for all): sum of all stock positions
        total_position = np.sum(self.portfolio_state[position_stock_idx, :])
        self.portfolio_state[position_portfolio_idx, :] = total_position

        # update cash left (same for all)
        self.portfolio_state[cash_left_idx, :] -= buy_amount_stock_i

        # update percentage of stock position: position stock/total balance
        self.portfolio_state[percentage_position_stock_idx, :] = self.portfolio_state[position_stock_idx,
                                                                 :] / self.portfolio_state[total_balance_idx, :]

        # update total balance: position portfolio + cash left
        self.portfolio_state[total_balance_idx, :] = total_position + self.portfolio_state[cash_left_idx, :]

        # update new stock shares: stock positions/ closing prices
        for i in range(NUM_STOCKS):
            self.portfolio_state[shares_stock_idx, i] = self.portfolio_state[position_stock_idx, i] / closing_prices[i]

        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'buy_0_25'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return buy_amount_stock_i

    def buy_0_50(self, closing_prices, stock_i, h, e, dates):

        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8,NUM_STOCKS)

        prev_balance = self.portfolio_state[total_balance_idx, 0]

        # change the value of position stock_i based on extra amount which is bought
        buy_amount_stock_i = 0.5 * h
        self.portfolio_state[position_stock_idx, stock_i] = self.portfolio_state[
                                                                position_stock_idx, stock_i] + buy_amount_stock_i

        # update position total (same for all): sum of all stock positions
        total_position = np.sum(self.portfolio_state[position_stock_idx, :])
        self.portfolio_state[position_portfolio_idx, :] = total_position

        # update cash left (same for all)
        self.portfolio_state[cash_left_idx, :] -= buy_amount_stock_i

        # update percentage of stock position: position stock/total balance
        self.portfolio_state[percentage_position_stock_idx, :] = self.portfolio_state[position_stock_idx,
                                                                 :] / self.portfolio_state[total_balance_idx, :]

        # update total balance: position portfolio + cash left
        self.portfolio_state[total_balance_idx, :] = total_position + self.portfolio_state[cash_left_idx, :]

        # update new stock shares: stock positions/ closing prices
        for i in range(NUM_STOCKS):
            self.portfolio_state[shares_stock_idx, i] = self.portfolio_state[position_stock_idx, i] / closing_prices[i]

        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'buy_0_50'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return buy_amount_stock_i

    def buy_0_75(self, closing_prices, stock_i, h, e, dates):

        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8,NUM_STOCKS)

        prev_balance = self.portfolio_state[total_balance_idx, 0]

        # change the value of position stock_i based on extra amount which is bought
        buy_amount_stock_i = 0.75 * h
        self.portfolio_state[position_stock_idx, stock_i] = self.portfolio_state[
                                                                position_stock_idx, stock_i] + buy_amount_stock_i

        # update position total (same for all): sum of all stock positions
        total_position = np.sum(self.portfolio_state[position_stock_idx, :])
        self.portfolio_state[position_portfolio_idx, :] = total_position

        # update cash left (same for all)
        self.portfolio_state[cash_left_idx, :] -= buy_amount_stock_i

        # update percentage of stock position: position stock/total balance
        self.portfolio_state[percentage_position_stock_idx, :] = self.portfolio_state[position_stock_idx,
                                                                 :] / self.portfolio_state[total_balance_idx, :]

        # update total balance: position portfolio + cash left
        self.portfolio_state[total_balance_idx, :] = total_position + self.portfolio_state[cash_left_idx, :]

        # update new stock shares: stock positions/ closing prices
        for i in range(NUM_STOCKS):
            self.portfolio_state[shares_stock_idx, i] = self.portfolio_state[position_stock_idx, i] / closing_prices[i]

        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'buy_0_75'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return buy_amount_stock_i

    def buy_1(self, closing_prices, stock_i, h, e, dates):

        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8,NUM_STOCKS)

        prev_balance = self.portfolio_state[total_balance_idx, 0]

        # change the value of position stock_i based on extra amount which is bought
        buy_amount_stock_i = 1 * h
        self.portfolio_state[position_stock_idx, stock_i] = self.portfolio_state[
                                                                position_stock_idx, stock_i] + buy_amount_stock_i

        # update position total (same for all): sum of all stock positions
        total_position = np.sum(self.portfolio_state[position_stock_idx, :])
        self.portfolio_state[position_portfolio_idx, :] = total_position

        # update cash left (same for all)
        self.portfolio_state[cash_left_idx, :] -= buy_amount_stock_i

        # update percentage of stock position: position stock/total balance
        self.portfolio_state[percentage_position_stock_idx, :] = self.portfolio_state[position_stock_idx,
                                                                 :] / self.portfolio_state[total_balance_idx, :]

        # update total balance: position portfolio + cash left
        self.portfolio_state[total_balance_idx, :] = total_position + self.portfolio_state[cash_left_idx, :]

        # update new stock shares: stock positions/ closing prices
        for i in range(NUM_STOCKS):
            self.portfolio_state[shares_stock_idx, i] = self.portfolio_state[position_stock_idx, i] / closing_prices[i]

        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'buy_1'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return buy_amount_stock_i

    def sell_0_1(self, closing_prices, stock_i, h, e, dates):

        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8,NUM_STOCKS)

        prev_balance = self.portfolio_state[total_balance_idx, 0]

        # change the value of position stock_i based on extra amount which is bought
        sell_amount_stock_i = 0.1 * h
        self.portfolio_state[position_stock_idx, stock_i] = self.portfolio_state[
                                                                position_stock_idx, stock_i] - sell_amount_stock_i

        # update position total (same for all): sum of all stock positions
        total_position = np.sum(self.portfolio_state[position_stock_idx, :])
        self.portfolio_state[position_portfolio_idx, :] = total_position

        # update cash left (same for all)
        self.portfolio_state[cash_left_idx, :] += sell_amount_stock_i

        # update percentage of stock position: position stock/total balance
        self.portfolio_state[percentage_position_stock_idx, :] = self.portfolio_state[position_stock_idx,
                                                                 :] / self.portfolio_state[total_balance_idx, :]

        # update total balance: position portfolio + cash left
        self.portfolio_state[total_balance_idx, :] = total_position + self.portfolio_state[cash_left_idx, :]

        # update new stock shares: stock positions/ closing prices
        for i in range(NUM_STOCKS):
            self.portfolio_state[shares_stock_idx, i] = self.portfolio_state[position_stock_idx, i] / closing_prices[i]

        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'sell_0_1'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return sell_amount_stock_i

    def sell_0_25(self, closing_prices, stock_i, h, e, dates):

        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8,NUM_STOCKS)

        prev_balance = self.portfolio_state[total_balance_idx, 0]

        # change the value of position stock_i based on extra amount which is bought
        sell_amount_stock_i = 0.25 * h
        self.portfolio_state[position_stock_idx, stock_i] = self.portfolio_state[
                                                                position_stock_idx, stock_i] - sell_amount_stock_i

        # update position total (same for all): sum of all stock positions
        total_position = np.sum(self.portfolio_state[position_stock_idx, :])
        self.portfolio_state[position_portfolio_idx, :] = total_position

        # update cash left (same for all)
        self.portfolio_state[cash_left_idx, :] += sell_amount_stock_i

        # update percentage of stock position: position stock/total balance
        self.portfolio_state[percentage_position_stock_idx, :] = self.portfolio_state[position_stock_idx,
                                                                 :] / self.portfolio_state[total_balance_idx, :]

        # update total balance: position portfolio + cash left
        self.portfolio_state[total_balance_idx, :] = total_position + self.portfolio_state[cash_left_idx, :]

        # update new stock shares: stock positions/ closing prices
        for i in range(NUM_STOCKS):
            self.portfolio_state[shares_stock_idx, i] = self.portfolio_state[position_stock_idx, i] / closing_prices[i]

        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'sell_0_25'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return sell_amount_stock_i

    def sell_0_50(self, closing_prices, stock_i, h, e, dates):

        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8,NUM_STOCKS)

        prev_balance = self.portfolio_state[total_balance_idx, 0]

        # change the value of position stock_i based on extra amount which is bought
        sell_amount_stock_i = 0.5 * h
        self.portfolio_state[position_stock_idx, stock_i] = self.portfolio_state[
                                                                position_stock_idx, stock_i] - sell_amount_stock_i

        # update position total (same for all): sum of all stock positions
        total_position = np.sum(self.portfolio_state[position_stock_idx, :])
        self.portfolio_state[position_portfolio_idx, :] = total_position

        # update cash left (same for all)
        self.portfolio_state[cash_left_idx, :] += sell_amount_stock_i

        # update percentage of stock position: position stock/total balance
        self.portfolio_state[percentage_position_stock_idx, :] = self.portfolio_state[position_stock_idx,
                                                                 :] / self.portfolio_state[total_balance_idx, :]

        # update total balance: position portfolio + cash left
        self.portfolio_state[total_balance_idx, :] = total_position + self.portfolio_state[cash_left_idx, :]

        # update new stock shares: stock positions/ closing prices
        for i in range(NUM_STOCKS):
            self.portfolio_state[shares_stock_idx, i] = self.portfolio_state[position_stock_idx, i] / closing_prices[i]

        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'sell_0_50'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return sell_amount_stock_i

    def sell_0_75(self, closing_prices, stock_i, h, e, dates):

        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8,NUM_STOCKS)

        prev_balance = self.portfolio_state[total_balance_idx, 0]

        # change the value of position stock_i based on extra amount which is bought
        sell_amount_stock_i = 0.75 * h
        self.portfolio_state[position_stock_idx, stock_i] = self.portfolio_state[
                                                                position_stock_idx, stock_i] - sell_amount_stock_i

        # update position total (same for all): sum of all stock positions
        total_position = np.sum(self.portfolio_state[position_stock_idx, :])
        self.portfolio_state[position_portfolio_idx, :] = total_position

        # update cash left (same for all)
        self.portfolio_state[cash_left_idx, :] += sell_amount_stock_i

        # update percentage of stock position: position stock/total balance
        self.portfolio_state[percentage_position_stock_idx, :] = self.portfolio_state[position_stock_idx,
                                                                 :] / self.portfolio_state[total_balance_idx, :]

        # update total balance: position portfolio + cash left
        self.portfolio_state[total_balance_idx, :] = total_position + self.portfolio_state[cash_left_idx, :]

        # update new stock shares: stock positions/ closing prices
        for i in range(NUM_STOCKS):
            self.portfolio_state[shares_stock_idx, i] = self.portfolio_state[position_stock_idx, i] / closing_prices[i]

        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'sell_0_75'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return sell_amount_stock_i

    def sell_1(self, closing_prices, stock_i, h, e, dates):

        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8,NUM_STOCKS)

        prev_balance = self.portfolio_state[total_balance_idx, 0]

        # change the value of position stock_i based on extra amount which is bought
        sell_amount_stock_i = 1 * h
        self.portfolio_state[position_stock_idx, stock_i] = self.portfolio_state[
                                                                position_stock_idx, stock_i] - sell_amount_stock_i

        # update position total (same for all): sum of all stock positions
        total_position = np.sum(self.portfolio_state[position_stock_idx, :])
        self.portfolio_state[position_portfolio_idx, :] = total_position

        # update cash left (same for all)
        self.portfolio_state[cash_left_idx, :] += sell_amount_stock_i

        # update percentage of stock position: position stock/total balance
        self.portfolio_state[percentage_position_stock_idx, :] = self.portfolio_state[position_stock_idx,
                                                                 :] / self.portfolio_state[total_balance_idx, :]

        # update total balance: position portfolio + cash left
        self.portfolio_state[total_balance_idx, :] = total_position + self.portfolio_state[cash_left_idx, :]

        # update new stock shares: stock positions/ closing prices
        for i in range(NUM_STOCKS):
            self.portfolio_state[shares_stock_idx, i] = self.portfolio_state[position_stock_idx, i] / closing_prices[i]

        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'sell_1'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return sell_amount_stock_i



    def hold(self, closing_prices, stock_i, h, e, dates):
        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8, num_stocks)

        amount_transactioned = 0.0
        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'hold'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return amount_transactioned

    def sell_everything(self, closing_prices, stock_i, h, e, dates):
        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8, num_stocks)

        # indexes
        # total_balance -> self.portfolio_state[total_balance_idx,:]
        # (cash) position per stock -> self.portfolio_state[position_stock_idx,:]
        # (cash) position portfolio -> self.portfolio_state[position_portfolio_idx,:]
        # daily return per stock -> self.portfolio_state[return_stock_idx,:]
        # daily return total portfolio -> self.portfolio_state[return_portfolio_idx,:]
        # cash left total -> self.portfolio_state[cash_left_idx,:]
        # percentage position per stock -> self.portfolio_state[percentage_position_stock_idx,:]
        # shares per stock -> self.portfolio_state[shares_stock_idx,:]

        # SELLING EVERYTHING ####################

        #  total balance after selling
        balance_sold = self.portfolio_state[total_balance_idx, 0]

        self.reset_portfolio()

        # update total balance: position plus cash left

        self.portfolio_state[total_balance_idx, :] = balance_sold

        # update cash left balance:
        self.portfolio_state[cash_left_idx, :] = balance_sold

        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'sell_everything'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return balance_sold

    def buy_1_share(self, closing_prices, stock_i, h, e, dates):

        # self.portfolio_state = np.asarray([self.total_balances,self.position_stocks,self.position_portfolio,
        #                                      self.daily_return_stock,self.daily_return_total,
        #                                      self.cash_left, self.percentage_positions, self.shares_per_stock])
        # shape: (8,NUM_STOCKS)

        prev_balance = self.portfolio_state[total_balance_idx, 0]

        # closing price is price per one share for stock_i
        buy_amount_stock_i = 1.0 * closing_prices[stock_i]

        # change the value of position stock_i based on extra amount which is bought
        self.portfolio_state[position_stock_idx, stock_i] = self.portfolio_state[
                                                                position_stock_idx, stock_i] + buy_amount_stock_i

        # update position total (same for all): sum of all stock positions
        total_position = np.sum(self.portfolio_state[position_stock_idx, :])
        self.portfolio_state[position_portfolio_idx, :] = total_position

        # update cash left (same for all)
        self.portfolio_state[cash_left_idx, :] -= buy_amount_stock_i

        # update percentage of stock position: position stock/total balance
        self.portfolio_state[percentage_position_stock_idx, :] = self.portfolio_state[position_stock_idx,
                                                                 :] / self.portfolio_state[total_balance_idx, :]

        # update total balance: position portfolio + cash left
        self.portfolio_state[total_balance_idx, :] = total_position + self.portfolio_state[cash_left_idx, :]

        # update new stock shares: stock positions/ closing prices
        for i in range(NUM_STOCKS):
            self.portfolio_state[shares_stock_idx, i] = self.portfolio_state[position_stock_idx, i] / closing_prices[i]

        # Explainability for last epoch
        if e == (self.num_epochs - 1):
            dates = dates
            actions = ['None'] * self.num_stocks
            actions[stock_i] = 'buy_1_share'
            tics = self.name_stocks_list
            rewards = [self.reward] * self.num_stocks

            dates_series = pd.Series(dates, name='Dates')
            closing_prices_series = pd.Series(closing_prices, name='Closing_Price')
            tics_series = pd.Series(tics, name='TIC')
            action_series = pd.Series(actions, name='Actions')
            rewards_series = pd.Series(rewards, name='Rewards_balance_return')

            # turning agent.portfolio_state to DataFrame
            df_portfolio_state = pd.DataFrame(self.portfolio_state, columns=self.name_stocks_list)
            df_portfolio_state.insert(0, 'TIC', self.portfolio_state_rows)
            df_portfolio_state.set_index('TIC', inplace=True)
            df_portfolio_state_T = df_portfolio_state.T.reset_index(drop=True)

            df_memory_step = pd.concat(
                [dates_series, tics_series, closing_prices_series, action_series, rewards_series, df_portfolio_state_T],
                axis=1)

            self.explainability_df = pd.concat([self.explainability_df, df_memory_step], ignore_index=True)

        return buy_amount_stock_i

    # MASKING ACTIONS BASED ON Q-output
    def maskActions(self, options, h, closing_prices, train=False):
        '''
        Function to mask/ disregard certain actions according to the portfolio of the agent.
        Params:
        options (troch.Tensor): (num_stocks x num_actions) tensor to be masked
        '''

        # actions_dict = {
        #     0: 'buy_0_1',
        #     1: 'buy_0_25',
        #     2: 'buy_0_50',
        #     3: 'buy_0_75',
        #     4: 'buy_1',
        #     5: 'sell_0_1',
        #     6: 'sell_0_25',
        #     7: 'sell_0_50',
        #     8: 'sell_0_75',
        #     9: 'sell_1',
        #     10: 'hold',
        #     11: 'sell_everything',
        #     12: 'buy_1_share'
        # }

        num_stocks = self.num_stocks
        num_actions = self.num_actions
        actions_dict = self.actions_dict
        device = self.device

        if train:
            options.requires_grad_(True)

        options_np = options.detach().cpu().numpy().reshape(num_stocks, num_actions).copy()
        stocks_options = []
        for stock_i in range(num_stocks):
            for action_idx in range(num_actions):
                if 'buy' in actions_dict[action_idx] and (0.1 * h) > self.portfolio_state[cash_left_idx, 0]:
                    options_np[stock_i, action_idx] = -100.0
                if 'buy' in actions_dict[action_idx] and self.portfolio_state[cash_left_idx, 0] <= 0.0:
                    options_np[stock_i, action_idx] = -100.0
                # normally buying options are not permitted if there is not enough cash left
                if 'buy_0_1' in actions_dict[action_idx] and 0.1 * h > self.portfolio_state[cash_left_idx, 0]:
                    options_np[stock_i, action_idx] = -100.0
                if 'buy_0_25' in actions_dict[action_idx] and 0.25 * h > self.portfolio_state[cash_left_idx, 0]:
                    options_np[stock_i, action_idx] = -100.0
                if 'buy_0_50' in actions_dict[action_idx] and 0.5 * h > self.portfolio_state[cash_left_idx, 0]:
                    options_np[stock_i, action_idx] = -100.0
                if 'buy_0_75' in actions_dict[action_idx] and 0.75 * h > self.portfolio_state[cash_left_idx, 0]:
                    options_np[stock_i, action_idx] = -100.0
                if 'buy_1' in actions_dict[action_idx] and 1.0 * h > self.portfolio_state[cash_left_idx, 0]:
                    options_np[stock_i, action_idx] = -100.0
                if 'buy_1_share' in actions_dict[action_idx] and closing_prices[stock_i] > self.portfolio_state[
                    cash_left_idx, 0]:
                    options_np[stock_i, action_idx] = -100.0
                # normally selling options are not permitted if the amount is nopt persent in the position for stock_i
                if 'sell_0_1' in actions_dict[action_idx] and 0.1 * h > self.portfolio_state[
                    position_stock_idx, stock_i]:
                    options_np[stock_i, action_idx] = -100.0
                if 'sell_0_25' in actions_dict[action_idx] and 0.25 * h > self.portfolio_state[
                    position_stock_idx, stock_i]:
                    options_np[stock_i, action_idx] = -100.0
                if 'sell_0_50' in actions_dict[action_idx] and 0.5 * h > self.portfolio_state[
                    position_stock_idx, stock_i]:
                    options_np[stock_i, action_idx] = -100.0
                if 'sell_0_75' in actions_dict[action_idx] and 0.75 * h > self.portfolio_state[
                    position_stock_idx, stock_i]:
                    options_np[stock_i, action_idx] = -100.0
                if 'sell_1' in actions_dict[action_idx] and 1.0 * h > self.portfolio_state[position_stock_idx, stock_i]:
                    options_np[stock_i, action_idx] = -100.0
                if 'sell_everything' in actions_dict[action_idx]:
                    # Check if all stock positions are 0.0
                    if np.all(self.portfolio_state[position_stock_idx, :] == 0.0):
                        options_np[stock_i, action_idx] = -100.0

        options = torch.from_numpy(options_np)
        options = torch.flatten(options)
        if train:
            options.requires_grad_(True)

        return options.to(device)