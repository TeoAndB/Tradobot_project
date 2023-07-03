import datetime
# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.config_model_DQN_return import *
from src.models.DQN_model_fin import Agent, Portfolio, getState, maskActions
#from functions import *

import logging
import sys
import time
import click
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Trains the FinRL model.
    """
    logger = logging.getLogger(__name__)

    data = pd.read_csv(f'{input_filepath}/{DATASET}')[:918]
    selected_data_entries = 'entries-0-til-918'
    print(f'Dataset used: {input_filepath}/{DATASET}')

    dataset_name = os.path.splitext(DATASET)[0]


    # Create an empty DataFrames for explainability
    cols_stocks = data['tic'].unique().tolist()

    # Set the seed for reproducibility
    random_seed = 42
    unique_dates = data['date'].unique().tolist()

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Calculate the number of samples for each split
    total_samples = len(unique_dates)
    num_train_samples = int(train_ratio * total_samples)
    num_val_samples = int(val_ratio * total_samples)

    # Split the data using slicing
    train_dates = unique_dates[:num_train_samples]
    validation_dates = unique_dates[num_train_samples:num_train_samples + num_val_samples]
    test_dates = unique_dates[num_train_samples + num_val_samples:]

    # Create the train, validation, and test DataFrames based on the selected dates
    train_data = data[data['date'].isin(train_dates)]
    validation_data = data[data['date'].isin(validation_dates)]
    test_data = data[data['date'].isin(test_dates)]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model running on {device}')
    h = hmax # user defined maximum amount to buy

    num_features = len(data.columns) + len(data.tic.unique()) - 1 #data_window after being preprocessed with one hot encoding

    e = NUM_EPOCHS-1 # for saving explainability and performing actions

    # RANDOM AGENT #########################################################################################################
    ########################################################################################################################
    EPSILON_DECAY = 1.0 #no decay
    EPSILON = 2.0 # no chance of randomness, as porbability is extracted from a unfirom distribution


    agent_random = Agent(num_stocks=NUM_STOCKS, actions_dict=ACTIONS_DICTIONARY, h=h, num_features=num_features, balance=INITIAL_AMOUNT, name_stocks=cols_stocks,
                  gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                  epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, tau=TAU, num_epochs=NUM_EPOCHS)

    action_index_arr_mask = np.arange(0, agent_random.num_actions * agent_random.num_stocks, 1, dtype=int).reshape(agent_random.num_stocks,
                                                                                                     agent_random.num_actions)

    # RUNNING BASELINE FOR VALIDATION PERIOD  #######################################################################

    unique_dates_validation = validation_data['date'].unique()

    epoch_numbers_history_validation = []
    cumulated_profits_list_validation = [INITIAL_AMOUNT]
    dates_validation = [unique_dates_validation[0]]

    agent_random.reset()
    agent_random.Q_network.eval()  # Set the model to evaluation mode
    agent_random.epsilon = 0.0  # no exploration
    unique_dates_validation = validation_data['date'].unique()

    l_validation = len(unique_dates_validation)

    for t in range(l_validation):
        ####
        data_window = validation_data.loc[(validation_data['date'] == unique_dates_validation[t])]

        # replace NaN values with 0.0
        data_window = data_window.fillna(0)
        dates = data_window['date'].tolist()

        state = getState(data_window, t, agent_random)

        closing_prices = data_window['close'].tolist()

        # take action a, observe reward and next_state
        agent_random.epsilon = 2.0

        action_index = agent_random.act(state, closing_prices)

        # Find the indeces (stock_i, action_index_for_stock_i) where action_index  is present in action_index_arr_mask
        indices = np.where(action_index_arr_mask == action_index)
        stock_i, action_index_for_stock_i = map(int, indices)

        reward = agent_random.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

        updated_balance = agent_random.portfolio_state[0, 0]
        cumulated_profits_list_validation.append(updated_balance)
        dates_validation.append(agent_random.timestamp_portfolio)

        # Next state should append the t+1 data and portfolio_state. It also updates the position of agent_random portfolio based on agent_random position
        next_state, agent_random = getState(data_window, t + 1, agent_random)
        state = next_state

        done = True if t == l_validation - 1 else False

    # printing portfolio state for validation at the end
    df_portfolio_state = pd.DataFrame(agent_random.portfolio_state, columns=cols_stocks)
    df_portfolio_state.insert(0, 'TIC', agent_random.portfolio_state_rows)
    print(f'Validation Period Random Agent: Portfolio state is \n: {df_portfolio_state}')

    # saving the explainability file
    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y-%m-%d_%H_%M")
    agent_random.explainability_df.to_csv(
        f'./reports/results_DQN/baseline_results/minute_frequency_data/random_agent_validation_period_{dataset_name}_{date_string}_{selected_data_entries}.csv', index=False)


    # RUNNING BASELINE FOR TESTING PERIOD  #######################################################################
    print('Validation Phase for Random Agent')
    agent_random.reset()
    agent_random.epsilon = 0.0 # no exploration
    agent_random.memory = []

    unique_dates_testing = test_data['date'].unique()
    cumulated_profits_list_testing = [INITIAL_AMOUNT]
    dates_testing = [unique_dates_testing[0]]
    e = agent_random.num_epochs-1 #for explainability

    l_testing = len(unique_dates_testing)
    for t in range(l_testing):
        ####
        data_window = test_data.loc[(test_data['date'] == unique_dates_testing[t])]

        # replace NaN values with 0.0
        data_window = data_window.fillna(0)
        dates = data_window['date'].tolist()

        state = getState(data_window, t, agent_random)

        closing_prices = data_window['close'].tolist()

        # take action a, observe reward and next_state
        agent_random.epsilon = 2.0
        action_index = agent_random.act(state, closing_prices)

        # Find the indeces (stock_i, action_index_for_stock_i) where action_index  is present in action_index_arr_mask
        indices = np.where(action_index_arr_mask == action_index)
        stock_i, action_index_for_stock_i = map(int, indices)

        reward = agent_random.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

        updated_balance = agent_random.portfolio_state[0, 0]
        cumulated_profits_list_testing.append(updated_balance)
        dates_testing.append(dates[0])

        # Next state should append the t+1 data and portfolio_state. It also updates the position of agent_random portfolio based on agent_random position
        next_state, agent_random = getState(data_window, t + 1, agent_random)
        state = next_state

        done = True if t == l_testing - 1 else False

    # printing portfolio state for testing at the end
    df_portfolio_state = pd.DataFrame(agent_random.portfolio_state, columns=cols_stocks)
    df_portfolio_state.insert(0, 'TIC', agent_random.portfolio_state_rows)
    print(f'Testing Period Random Agent: Portfolio state is \n: {df_portfolio_state}')

    # saving the explainability file
    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y-%m-%d_%H_%M")
    agent_random.explainability_df.to_csv(
        f'./reports/results_DQN/baseline_results/minute_frequency_data/random_agent_testing_period_{dataset_name}_{date_string}_{selected_data_entries}.csv', index=False)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot for testing data
    ax1.plot(dates_validation, cumulated_profits_list_validation)
    ax1.set_title("Random Agent: Cumulated Profits Over Time (Validation Period))")
    ax1.set_xlabel("Dates")
    ax1.set_ylabel("Cumulated Profits")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)

    # Reduce the number of dates shown on the x-axis for testing data
    num_dates = 6
    skip = max(1, len(dates_validation) // num_dates)
    ax1.set_xticks(range(0, len(dates_validation), skip))
    ax1.set_xticklabels(dates_validation[::skip])
    ax1.set_xlim(0, len(dates_validation))
    ax1.tick_params(axis='x', labelsize=8)
    fig.autofmt_xdate(bottom=0.2)

    # Plot for validation data
    ax2.plot(dates_testing, cumulated_profits_list_testing)
    ax2.set_title("Random Agent: Cumulated Profits Over Time (Testing Period)")
    ax2.set_xlabel("Dates")
    ax2.set_ylabel("Cumulated Profits")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True)

    # Reduce the number of dates shown on the x-axis for validation data
    num_dates = 6
    skip = max(1, len(dates_testing) // num_dates)
    ax2.set_xticks(range(0, len(dates_testing), skip))
    ax2.set_xticklabels(dates_testing[::skip])
    ax2.set_xlim(0, len(dates_testing))
    ax2.tick_params(axis='x', labelsize=8)
    fig.autofmt_xdate(bottom=0.2)

    # Save the figure in the specified folder path
    plt.savefig(
        f'./reports/figures/baseline_models/minute_frequency_data/random_agent_testing_and_validation_periods_profits_for_{dataset_name}_{date_string}_{selected_data_entries}.png')

    # Show the figures
    plt.show()

    # BALANCED AGENT #########################################################################################################
    ##########################################################################################################################
    # divides the investing amount equally between stocks and holds ##########################################################


    agent_balanced = Agent(num_stocks=NUM_STOCKS, actions_dict=ACTIONS_DICTIONARY, h=h, num_features=num_features, balance=INITIAL_AMOUNT, name_stocks=cols_stocks,
                  gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                  epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, tau=TAU, num_epochs=NUM_EPOCHS)

    action_index_arr_mask = np.arange(0, agent_balanced.num_actions * agent_balanced.num_stocks, 1, dtype=int).reshape(agent_balanced.num_stocks,
                                                                                                     agent_balanced.num_actions)

    # RUNNING BASELINE FOR VALIDATION PERIOD  #######################################################################

    unique_dates_validation = validation_data['date'].unique()

    cumulated_profits_list_validation = [INITIAL_AMOUNT]
    dates_validation = [unique_dates_validation[0]]

    agent_balanced.reset()
    agent_balanced.Q_network.eval()  # Set the model to evaluation mode
    agent_balanced.epsilon = 0.0  # no exploration

    l_validation = len(unique_dates_validation)

    # at the beginning, invest equal amount in the same stock
    equal_amount = np.floor(INITIAL_AMOUNT / agent_balanced.num_stocks)

    for t in range(l_validation):
        ####
        data_window = validation_data.loc[(validation_data['date'] == unique_dates_validation[t])]

        # replace NaN values with 0.0
        data_window = data_window.fillna(0)
        dates = data_window['date'].tolist()

        state = getState(data_window, t, agent_balanced)

        closing_prices = data_window['close'].tolist()


        if t==0:
            for stock_i in range(agent_balanced.num_stocks):
                reward = agent_balanced.buy_1(closing_prices, stock_i, equal_amount, e, dates)

        else:
            stock_i = 0 # doesn't matter which stock
            action_index_for_stock_i = 10 #is the hold action

            reward = agent_balanced.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

        updated_balance = agent_balanced.portfolio_state[0, 0]
        cumulated_profits_list_validation.append(updated_balance)
        dates_validation.append(agent_balanced.timestamp_portfolio)

        # Next state should append the t+1 data and portfolio_state. It also updates the position of agent_random portfolio based on agent_random position
        next_state, agent_random = getState(data_window, t + 1, agent_balanced)
        state = next_state

        done = True if t == l_validation - 1 else False

    # printing portfolio state for validation at the end
    df_portfolio_state = pd.DataFrame(agent_balanced.portfolio_state, columns=cols_stocks)
    df_portfolio_state.insert(0, 'TIC', agent_balanced.portfolio_state_rows)
    print(f'Validation Period Balanced Agent: Portfolio state is \n: {df_portfolio_state}')

    # saving the explainability file
    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y-%m-%d_%H_%M")
    agent_balanced.explainability_df.to_csv(
        f'./reports/results_DQN/baseline_results/minute_frequency_data/balanced_agent_validation_period_{dataset_name}_{date_string}_{selected_data_entries}.csv', index=False)


    # RUNNING BASELINE FOR TESTING PERIOD  #######################################################################
    print('Testing Phase')
    unique_dates_testing = test_data['date'].unique()

    cumulated_profits_list_testing = [INITIAL_AMOUNT, INITIAL_AMOUNT]
    dates_testing = [unique_dates_testing[0], unique_dates_testing[0]]
    e = agent_balanced.num_epochs-1 #for explainability

    agent_balanced.reset()
    agent_balanced.epsilon = 0.0 # no exploration
    agent_balanced.memory = []

    # at the beginning, invest equal amount in the same stock
    equal_amount = np.floor(INITIAL_AMOUNT / agent_balanced.num_stocks)

    l_testing = len(unique_dates_testing)
    for t in range(l_testing):
        ####
        data_window = test_data.loc[(test_data['date'] == unique_dates_testing[t])]

        # replace NaN values with 0.0
        data_window = data_window.fillna(0)
        dates = data_window['date'].tolist()

        state = getState(data_window, t, agent_random)

        closing_prices = data_window['close'].tolist()

        if t == 0:
            for stock_i in range(agent_balanced.num_stocks):
                reward = agent_balanced.buy_1(closing_prices, stock_i, equal_amount, e, dates)

        else:
            stock_i = 0  # doesn't matter which stock
            action_index_for_stock_i = 10  # is the hold action

            reward = agent_balanced.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

        updated_balance = agent_balanced.portfolio_state[0, 0]
        cumulated_profits_list_testing.append(updated_balance)
        dates_testing.append(dates[0])

        # Next state should append the t+1 data and portfolio_state. It also updates the position of agent_random portfolio based on agent_random position
        next_state, agent_balanced = getState(data_window, t + 1, agent_random)
        state = next_state

        done = True if t == l_testing - 1 else False

    # printing portfolio state for testing at the end
    df_portfolio_state = pd.DataFrame(agent_balanced.portfolio_state, columns=cols_stocks)
    df_portfolio_state.insert(0, 'TIC', agent_balanced.portfolio_state_rows)
    print(f'Testing Period Balanced Agent: Portfolio state is \n: {df_portfolio_state}')

    # saving the explainability file
    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y-%m-%d_%H_%M")
    agent_balanced.explainability_df.to_csv(
        f'./reports/results_DQN/baseline_results/minute_frequency_data/balanced_agent_testing_period_{dataset_name}_{date_string}_{selected_data_entries}.csv', index=False)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot for testing data
    ax1.plot(dates_validation, cumulated_profits_list_validation)
    ax1.set_title("Balanced Agent: Cumulated Profits Over Time (Validation Period)")
    ax1.set_xlabel("Dates")
    ax1.set_ylabel("Cumulated Profits")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)

    # Reduce the number of dates shown on the x-axis for testing data
    num_dates = 6
    skip = max(1, len(dates_validation) // num_dates)
    ax1.set_xticks(range(0, len(dates_validation), skip))
    ax1.set_xticklabels(dates_validation[::skip])
    ax1.set_xlim(0, len(dates_validation))
    ax1.tick_params(axis='x', labelsize=8)
    fig.autofmt_xdate(bottom=0.2)

    # Plot for validation data
    ax2.plot(dates_testing, cumulated_profits_list_testing)
    ax2.set_title("Balanced Agent: Cumulated Profits Over Time (Testing Period)")
    ax2.set_xlabel("Dates")
    ax2.set_ylabel("Cumulated Profits")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True)

    # Reduce the number of dates shown on the x-axis for validation data
    num_dates = 6
    skip = max(1, len(dates_testing) // num_dates)
    ax2.set_xticks(range(0, len(dates_testing), skip))
    ax2.set_xticklabels(dates_testing[::skip])
    ax2.set_xlim(0, len(dates_testing))
    ax2.tick_params(axis='x', labelsize=8)
    fig.autofmt_xdate(bottom=0.2)

    # Save the figure in the specified folder path
    plt.savefig(
        f'./reports/figures/baseline_models/minute_frequency_data/balanced_agent_testing_and_validation_periods_profits_for_{dataset_name}_{date_string}_{selected_data_entries}.png')

    # Show the figures
    plt.show()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

