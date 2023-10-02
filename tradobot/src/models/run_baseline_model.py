import datetime
# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.config_model_DQN_return import *
from src.models.DQN_model_w_return_simple import Agent, Portfolio, getState
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

    h = np.floor(INITIAL_AMOUNT/NUM_STOCKS)

    data = pd.read_csv(f'{input_filepath}/{DATASET}')

    print(f'Dataset used: {input_filepath}/{DATASET}')

    dataset_name = os.path.splitext(DATASET)[0]

    data_type = 'daily_frequency_data'
    # data_type = 'minute_frequency_data'
    selected_adjustments = 'reset_portfolio_each_year'

    # Create an empty DataFrames for explainability
    cols_stocks = data['tic'].unique().tolist()

    # Set the seed for reproducibility
    random_seed = 42
    unique_dates = data['date'].unique().tolist()

    # Define the desired date ranges
    train_start_date = '2014-06-01'
    train_end_date = '2019-12-31'
    final_training_year = '2019'

    val_start_date = '2020-01-01'
    val_end_date = '2022-12-31'
    final_validation_year = '2021'

    test_start_date = '2022-01-01'
    test_end_date = '2022-12-31'
    final_testing_year = '2022'

    # Filter data based on date ranges
    training_data = data[(data['date'] >= train_start_date) & (data['date'] <= train_end_date)]
    validation_data = data[(data['date'] >= val_start_date) & (data['date'] <= val_end_date)]
    test_data = data[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]
    test_data2 = data[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model running on {device}')
    h = hmax # user defined maximum amount to buy

    num_features = len(data.columns) + len(data.tic.unique()) - 1 #data_window after being preprocessed with one hot encoding

    agent_balanced = Agent(num_stocks=NUM_STOCKS, actions_dict=ACTIONS_DICTIONARY, h=h, num_features=num_features, balance=INITIAL_AMOUNT, name_stocks=cols_stocks,
                  gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                  epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, tau=TAU, num_epochs=NUM_EPOCHS)

    action_index_arr_mask = np.arange(0, agent_balanced.num_actions * agent_balanced.num_stocks, 1, dtype=int).reshape(agent_balanced.num_stocks,
                                                                                                     agent_balanced.num_actions)

    # RUNNING BASELINE FOR TRAINING PERIOD  #######################################################################

    unique_dates_training = training_data['date'].unique()

    cumulated_profits_list_training = [INITIAL_AMOUNT]
    dates_training = [f'{final_training_year}-01-01']

    agent_balanced.reset()
    agent_balanced.Q_network.eval()  # Set the model to evaluation mode
    agent_balanced.epsilon = 0.0  # no exploration

    l_training = len(unique_dates_training)

    # at the beginning, invest equal amount in the same stock
    equal_amount = np.floor(INITIAL_AMOUNT / agent_balanced.num_stocks)

    years_list_training = []
    yearly_balance_training = []

    timestamps_list_training = [f'{final_training_year}-01-01']
    timestamps_list_validation = [f'{final_validation_year}-01-01']

    e = NUM_EPOCHS-1 # for saving explainability and performing actions

    for t in range(l_training):
        ####
        data_window = training_data.loc[(training_data['date'] == unique_dates_training[t])]

        # replace NaN values with 0.0
        data_window = data_window.fillna(0)
        dates = data_window['date'].tolist()

        # state = getState(data_window, t, agent_balanced)

        closing_prices = data_window['close'].tolist()

        if t < (l_training - 1):
            next_data_window = training_data.loc[(training_data['date'] == unique_dates_training[t + 1])]
            # next_state = getState(next_data_window, t + 1, agent_balanced)
            next_closing_prices = next_data_window['close'].tolist()
            next_dates = next_data_window['date'].tolist()
        else:
            # next_state = state
            next_closing_prices = closing_prices
            next_dates = dates

        # Initially buy equal shares of all stocks, then hold
        if t == 0:
            for stock_i in range(agent_balanced.num_stocks):
                action_index_for_stock_i = 4  # is the buy 1 share action

                amount_transaction = agent_balanced.execute_action(action_index_for_stock_i, closing_prices, stock_i, h,
                                                                   e, dates)

            reward = agent_balanced.update_portfolio(next_closing_prices, next_dates, stock_i, amount_transaction)
            agent_balanced.reward = reward

        else:
            stock_i = 0  # doesn't matter which stock
            action_hold_stock_i = 10  # is the hold action

            amount_transaction = agent_balanced.execute_action(action_hold_stock_i, closing_prices, stock_i, h, e, dates)
            reward = agent_balanced.update_portfolio(next_closing_prices, next_dates, stock_i, amount_transaction)
            agent_balanced.reward = reward

        done = True if t == l_training - 1 else False

        # extract the years
        data_window = data_window.copy()
        data_window['date'] = pd.to_datetime(data_window['date'])
        data_window['date'] = pd.to_datetime(data_window['date'])
        years = data_window['date'].dt.strftime('%Y')
        year = years.iloc[0]

        next_data_window = next_data_window.copy()
        next_data_window['date'] = pd.to_datetime(next_data_window['date'])
        next_data_window['date'] = pd.to_datetime(next_data_window['date'])
        next_years = next_data_window['date'].dt.strftime('%Y')
        next_year = next_years.iloc[0]

        if year != next_year:
            years_list_training.append(year)
            yearly_balance_training.append(agent_balanced.portfolio_state[0, 0])
            agent_balanced.reset_portfolio()

            # re-buy equal amount of shares
            for stock_i in range(agent_balanced.num_stocks):
                reward = agent_balanced.buy_1(closing_prices, stock_i, equal_amount, e, dates)
            reward = agent_balanced.update_portfolio(next_closing_prices, next_dates, stock_i, amount_transaction)
            agent_balanced.reward = reward

        # track profits only from the most recent year:
        if year == final_training_year:
            cumulated_profits_list_training.append(agent_balanced.portfolio_state[0, 0])
            dates_training.append(agent_balanced.timestamp_portfolio)

        # note the last profit at the end of the final year
        if done:
            # keep track of yearly profit
            years_list_training.append(year)
            yearly_balance_training.append(agent_balanced.portfolio_state[0, 0])


    # printing portfolio state for training at the end
    df_portfolio_state = pd.DataFrame(agent_balanced.portfolio_state, columns=cols_stocks)
    df_portfolio_state.insert(0, 'TIC', agent_balanced.portfolio_state_rows)
    print(f'Training Period Balanced Agent: Portfolio state is \n: {df_portfolio_state}')

    # saving the explainability file
    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y-%m-%d_%H_%M") + 'validation_noreset'
    agent_balanced.explainability_df.to_csv(
        f'./reports/results_DQN/baseline_results/minute_frequency_data/balanced_agent_training_period_{dataset_name}_{date_string}_{selected_adjustments}.csv',
        index=False)

    # RUNNING BASELINE FOR VALIDATION PERIOD  #######################################################################

    unique_dates_validation = validation_data['date'].unique()

    cumulated_profits_list_validation = [INITIAL_AMOUNT]
    dates_validation = [f'{final_validation_year}-01-01']

    agent_balanced.reset()
    agent_balanced.reset_portfolio()
    agent_balanced.Q_network.eval()  # Set the model to evaluation mode
    agent_balanced.epsilon = 0.0  # no exploration

    l_validation = len(unique_dates_validation)

    # at the beginning, invest equal amount in the same stock
    equal_amount = np.floor(INITIAL_AMOUNT / agent_balanced.num_stocks)
    yearly_balance_validation = []
    years_list_validation = []

    for t in range(l_validation):
        ####
        data_window = validation_data.loc[(validation_data['date'] == unique_dates_validation[t])]

        # replace NaN values with 0.0
        data_window = data_window.fillna(0)
        dates = data_window['date'].tolist()

        # state = getState(data_window, t, agent_balanced)

        closing_prices = data_window['close'].tolist()

        if t < (l_validation - 1):
            next_data_window = validation_data.loc[(validation_data['date'] == unique_dates_validation[t + 1])]
            # next_state = getState(next_data_window, t + 1, agent_balanced)
            next_closing_prices = next_data_window['close'].tolist()
            next_dates = next_data_window['date'].tolist()
        else:
            # next_state = state
            next_closing_prices = closing_prices
            next_dates = dates

        if t==0:
            for stock_i in range(agent_balanced.num_stocks):
                action_index_for_stock_i = 4  # is the buy 1 share action

                amount_transaction = agent_balanced.execute_action(action_index_for_stock_i, closing_prices, stock_i, h,
                                                                   e, dates)

            reward = agent_balanced.update_portfolio(next_closing_prices, next_dates, stock_i, amount_transaction)
            agent_balanced.reward = reward

        else:
            stock_i = 0 # doesn't matter which stock
            action_hold_stock_i = 10 #is the hold action

            amount_transaction = agent_balanced.execute_action(action_hold_stock_i, closing_prices, stock_i, h, e, dates)
            reward = agent_balanced.update_portfolio(next_closing_prices, next_dates, stock_i, amount_transaction)
            agent_balanced.reward = reward

        done = True if t == l_validation - 1 else False

        # extract the years
        data_window = data_window.copy()
        data_window['date'] = pd.to_datetime(data_window['date'])
        data_window['date'] = pd.to_datetime(data_window['date'])
        years = data_window['date'].dt.strftime('%Y')
        year = years.iloc[0]

        next_data_window = next_data_window.copy()
        next_data_window['date'] = pd.to_datetime(next_data_window['date'])
        next_data_window['date'] = pd.to_datetime(next_data_window['date'])
        next_years = next_data_window['date'].dt.strftime('%Y')
        next_year = next_years.iloc[0]

        if year != next_year:

            years_list_validation.append(year)
            yearly_balance_validation.append(agent_balanced.portfolio_state[0, 0])
            agent_balanced.reset_portfolio()

            # re-buy equal amount
            for stock_i in range(agent_balanced.num_stocks):
                reward = agent_balanced.buy_1(closing_prices, stock_i, equal_amount, e, dates)

            reward = agent_balanced.update_portfolio(next_closing_prices, next_dates, stock_i, amount_transaction)
            agent_balanced.reward = reward

        # track profits only from the most recent year:
        if year == final_validation_year:
            cumulated_profit_per_epoch = agent_balanced.portfolio_state[0, 0]
            cumulated_profits_list_validation.append(cumulated_profit_per_epoch)
            dates_validation.append(agent_balanced.timestamp_portfolio)

        # note the last profit at the end of the final year
        if done:
            # keep track of yearly profit
            years_list_validation.append(year)
            yearly_balance_validation.append(agent_balanced.portfolio_state[0, 0])

    # printing portfolio state for validation at the end
    df_portfolio_state = pd.DataFrame(agent_balanced.portfolio_state, columns=cols_stocks)
    df_portfolio_state.insert(0, 'TIC', agent_balanced.portfolio_state_rows)
    print(f'Validation Period Balanced Agent: Portfolio state is \n: {df_portfolio_state}')

    # saving the explainability file
    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y-%m-%d_%H_%M")
    agent_balanced.explainability_df.to_csv(
        f'./reports/results_DQN/baseline_results/minute_frequency_data/balanced_agent_validation_period_{dataset_name}_{date_string}_{selected_adjustments}.csv', index=False)


    # RUNNING BASELINE FOR TESTING PERIOD 1  #######################################################################
    print('Testing Phase')
    unique_dates_testing = test_data['date'].unique()

    cumulated_profits_list_testing = [INITIAL_AMOUNT]
    dates_testing = [f'{final_testing_year}-01-01']
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

        # state = getState(data_window, t, agent_balanced)

        closing_prices = data_window['close'].tolist()

        if t < (l_testing - 1):
            next_data_window = test_data.loc[(test_data['date'] == unique_dates_testing[t + 1])]
            next_closing_prices = next_data_window['close'].tolist()
            next_dates = next_data_window['date'].tolist()
        else:
            # next_state = state
            next_closing_prices = closing_prices
            next_dates = dates

        if t == 0:
            for stock_i in range(agent_balanced.num_stocks):
                action_index_for_stock_i = 4  # is the buy 1h

                amount_transaction = agent_balanced.execute_action(action_index_for_stock_i, closing_prices, stock_i, h,
                                                                   e, dates)

            reward = agent_balanced.update_portfolio(next_closing_prices, next_dates, stock_i, amount_transaction)
            agent_balanced.reward = reward

        else:
            stock_i = 0  # doesn't matter which stock
            action_index_for_stock_i = 10  # is the hold action

            amount_transaction = agent_balanced.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

            reward = agent_balanced.update_portfolio(next_closing_prices, next_dates, stock_i, amount_transaction)
            agent_balanced.reward = reward

        cumulated_profits_list_testing.append(agent_balanced.portfolio_state[0,0])
        dates_testing.append(agent_balanced.timestamp_portfolio)

        done = True if t == l_testing - 1 else False


    # printing portfolio state for testing at the end
    df_portfolio_state = pd.DataFrame(agent_balanced.portfolio_state, columns=cols_stocks)
    df_portfolio_state.insert(0, 'TIC', agent_balanced.portfolio_state_rows)
    print(f'Testing Period Balanced Agent: Portfolio state is \n: {df_portfolio_state}')

    # saving the explainability file
    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y-%m-%d_%H_%M")
    agent_balanced.explainability_df.to_csv(
        f'./reports/results_DQN/baseline_results/minute_frequency_data/balanced_agent_testing_period_{dataset_name}_{date_string}_{selected_adjustments}.csv', index=False)

    # Plotting
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(30, 6))  # Adjusted for three subplots

    # Plot for training data
    ax0.plot(dates_training, cumulated_profits_list_training)
    ax0.set_title(f'Balanced Agent: Monetary Balance during Training - year {final_training_year})')
    ax0.set_xlabel("Dates")
    ax0.set_ylabel("Monetary Balance")
    ax0.tick_params(axis='x', rotation=45)
    ax0.grid(True)

    # Reduce the number of dates shown on the x-axis for training data
    num_dates = 6
    skip = max(1, len(dates_training) // num_dates)
    ax0.set_xticks(range(0, len(dates_training), skip))
    ax0.set_xticklabels(dates_training[::skip])
    ax0.set_xlim(0, len(dates_training))
    ax0.tick_params(axis='x', labelsize=8)

    # Plot for validation data
    ax1.plot(dates_validation, cumulated_profits_list_validation)
    ax1.set_title(f'Balanced Agent: Monetary Balance during Evaluation - year {final_validation_year})')
    ax1.set_xlabel("Dates")
    ax1.set_ylabel("Monetary Balance")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)

    # Reduce the number of dates shown on the x-axis for validation data
    skip = max(1, len(dates_validation) // num_dates)
    ax1.set_xticks(range(0, len(dates_validation), skip))
    ax1.set_xticklabels(dates_validation[::skip])
    ax1.set_xlim(0, len(dates_validation))
    ax1.tick_params(axis='x', labelsize=8)

    # Plot for testing data
    ax2.plot(dates_testing, cumulated_profits_list_testing)
    ax2.set_title(f'Balanced Agent: Monetary Balance during Testing - year {final_testing_year})')
    ax2.set_xlabel("Dates")
    ax2.set_ylabel("Monetary Balance")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True)

    # Reduce the number of dates shown on the x-axis for testing data
    skip = max(1, len(dates_testing) // num_dates)
    ax2.set_xticks(range(0, len(dates_testing), skip))
    ax2.set_xticklabels(dates_testing[::skip])
    ax2.set_xlim(0, len(dates_testing))
    ax2.tick_params(axis='x', labelsize=8)

    fig.autofmt_xdate(bottom=0.2)
    plt.tight_layout(pad=5.0)

    # Save the figure in the specified folder path
    plt.savefig(
        f'./reports/figures/baseline_models/{data_type}/balanced_agent_testing_and_validation_periods_profits_for_{dataset_name}_{date_string}_{selected_adjustments}.png')

    # Show the figures
    plt.show()

    # Average profit for training and validation
    avg_profit_training = sum(yearly_balance_training)/len(yearly_balance_training)
    avg_profit_validation = sum(yearly_balance_validation)/len(yearly_balance_validation)
    avg_profit_testing = cumulated_profits_list_testing[-1]
    avg_cumulated_balance = [avg_profit_training, avg_profit_validation, avg_profit_testing]
    avg_yearly_profit = [avg_profit_training/INITIAL_AMOUNT, avg_profit_validation/INITIAL_AMOUNT, avg_profit_testing/INITIAL_AMOUNT]
    dataset_types = ['training','validation','testing']

    avg_yearly_balance_df = pd.DataFrame(
        {'dataset': dataset_types,
         'avg_yearly_balance': avg_cumulated_balance,
         'avg_yearly_profit': avg_yearly_profit
         })

    avg_yearly_balance_df.to_csv(
        f'./reports/tables/results_DQN/baseline_results/{data_type}/avg_yearly_balance_{dataset_name}_{date_string}.csv', index=False)

    print(f'Baseline Model: Overall Average yearly profit \n{avg_yearly_balance_df}')

    
    # Profit per year
    profit_per_year_last_epoch_training = pd.DataFrame(
        {'year_training': years_list_training,
         'balance_per_year_training': yearly_balance_training,
         'yearly_profit_training': [balance/INITIAL_AMOUNT for balance in yearly_balance_training]
         })

    profit_per_year_last_epoch_validation = pd.DataFrame(
        {
         'year_validation': years_list_validation,
         'balance_per_year_validation': yearly_balance_validation,
         'yearly_profit_validation': [balance/INITIAL_AMOUNT for balance in yearly_balance_validation]
         })

    profit_per_year_last_epoch_training.to_csv(
        f'./reports/tables/results_DQN/baseline_results/{data_type}/profit_per_year_last_epoch_training_{dataset_name}_{date_string}.csv', index=False)

    print(f'Baseline Model: Profit per Year for Last Epoch - Training \n{profit_per_year_last_epoch_training}')


    profit_per_year_last_epoch_validation.to_csv(
        f'./reports/tables/results_DQN/baseline_results/{data_type}/profit_per_year_last_epoch_validation_{dataset_name}_{date_string}.csv', index=False)

    print(f'Baseline Model: Profit per Year for Last Epoch - Validation \n{profit_per_year_last_epoch_validation}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

