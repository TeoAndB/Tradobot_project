import datetime
# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.config_model_DQN_return import *
from src.models.DQN_model_w_return import Agent, Portfolio, getState
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
from matplotlib.ticker import MaxNLocator

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Trains the FinRL model.
    """
    logger = logging.getLogger(__name__)

    data = pd.read_csv(f'{input_filepath}/{DATASET}', index_col=0)
    #data = normalize_dataframe(data.copy())

    selected_data_entries = 'HuberLoss_L1L2reg_wLSTM_article_dates'
    #data_type = "minute_frequency_data"
    data_type = "daily_frequency_data"
    reward_type = "reward_portfolio_return"

    print(f'Dataset used: {input_filepath}/{DATASET}')

    dataset_name = os.path.splitext(DATASET)[0]


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
    val_end_date = '2021-12-31'
    final_validation_year = '2021'

    # extra insights for year 2022
    test_start_date = '2022-01-01'
    test_end_date = '2022-12-31'
    final_testing_year = '2022'

    # extra insights for year 2021
    test_start_date_2 = '2021-01-01'
    test_end_date_2 = '2021-12-31'
    final_testing_year_2 = '2021'

    # Filter data based on date ranges
    train_data = data[(data['date'] >= train_start_date) & (data['date'] <= train_end_date)]
    validation_data = data[(data['date'] >= val_start_date) & (data['date'] <= val_end_date)]
    test_data = data[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]
    test_data_2 = data[(data['date'] >= test_start_date_2) & (data['date'] <= test_end_date_2)]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model running on {device}')
    h = hmax # user defined maximum amount to buy

    num_features = len(data.columns) + len(data.tic.unique()) - 1 #data_window after being preprocessed with one hot encoding

    agent = Agent(num_stocks=NUM_STOCKS, actions_dict=ACTIONS_DICTIONARY, h=h, num_features=num_features, balance=INITIAL_AMOUNT, name_stocks=cols_stocks,
                  gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                  epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, tau=TAU, num_epochs=NUM_EPOCHS)

    action_index_arr_mask = np.arange(0, agent.num_actions * agent.num_stocks, 1, dtype=int).reshape(agent.num_stocks,
                                                                                                     agent.num_actions)

    # TRAINING lists for keeping track of losses and profits #############################################################
    unique_dates_training = train_data['date'].unique()

    train_loss_history = []
    loss_history_per_epoch_training = []
    epoch_numbers_history_training = []
    cumulated_profits_list_training = [INITIAL_AMOUNT]
    cumulated_profits_list_training_per_epcoh_list = [INITIAL_AMOUNT]
    epoch_numbers_history_training_for_profits = [0]
    timestamps_list_training = [f'{final_training_year}-01-01']

    # VALIDATION lists for keeping track of losses and profits ###########################################################

    unique_dates_validation = validation_data['date'].unique()

    val_loss_history = []
    loss_history_per_epoch_validation = []
    epoch_numbers_history_validation = []
    cumulated_profits_list_validation = [INITIAL_AMOUNT]
    cumulated_profits_list_validation_per_epcoh_list = [INITIAL_AMOUNT]
    epoch_numbers_history_val_for_profits = [0]
    epoch_numbers_history_validation_loss = []

    timestamps_list_validation = [f'{final_validation_year}-01-01']

    epoch_numbers_history_training_loss = []
    for e in range(agent.num_epochs):
        print(f'Epoch {e+1}')
        data_window = train_data.loc[(train_data['date'] == unique_dates_training[0])]
        initial_state = getState(data_window, 0, agent)
        closing_prices = data_window['close'].tolist()

        cumulated_profits_list_training = [INITIAL_AMOUNT]
        cumulated_profits_list_validation = [INITIAL_AMOUNT]

        agent.reset()

        # TRAINING PHASE ##################################################################

        l_training = len(unique_dates_training)
        max_profit = 0.0
        loss_per_50_timesteps = []
        yearly_profit_training = []
        final_penalty = 1.1
        years_list_training = []

        for t in range(l_training):

            data_window = train_data.loc[(train_data['date'] == unique_dates_training[t])]

            # replace NaN values with 0.0
            data_window = data_window.fillna(0)
            # get the date
            dates = data_window['date'].tolist()
            # get the year for assuming yearly profits
            data_window = data_window.copy()
            data_window['date'] = pd.to_datetime(data_window['date'])
            years = data_window['date'].dt.strftime('%Y')
            year = years.iloc[0]

            state = getState(data_window, t, agent)

            # take action a, observe reward and next_state
            closing_prices = data_window['close'].tolist()

            action_index = agent.act(state, closing_prices)

            # Find the indeces (stock_i, action_index_for_stock_i) where action_index  is present in action_index_arr_mask

            indices = np.where(action_index_arr_mask == action_index)
            stock_i, action_index_for_stock_i = map(int, indices)

            agent.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

            # Next state should append the t+1 data and portfolio_state. It also updates the position of agent portfolio based on agent position

            if t<(l_training-1):
                next_data_window = train_data.loc[(train_data['date'] == unique_dates_training[t + 1])]
                next_state = getState(next_data_window, t+1, agent)
                next_closing_prices = next_data_window['close'].tolist()
                next_dates = next_data_window['date'].tolist()
            else:
                next_state = state
                next_closing_prices = closing_prices
                next_dates = dates

            reward = agent.update_portfolio(next_closing_prices, next_dates, stock_i)

            agent.reward = reward * final_penalty
            if final_penalty != 1.0:
                final_penalty = 1.0

            done = True if t == l_training - 1 else False

            agent.remember(state=state, actions=(action_index, action_index_for_stock_i, stock_i, h), closing_prices=closing_prices,
                           reward=reward, next_state=next_state, done=done)

            if len(agent.memory) >= agent.batch_size:
                agent.expReplay(e) # will also save the model on the last epoch

                batch_loss_history = agent.batch_loss_history.copy()
                train_loss_history.extend(batch_loss_history)


            if (t%50 == 0 or t==(l_training-1)) and len(train_loss_history)>0:
                loss_per_epoch_log = sum(train_loss_history) / len(train_loss_history)
                loss_per_50_timesteps.append(loss_per_epoch_log)
                print(f'Epoch {e+1}, Training Loss: {loss_per_epoch_log:.4f}')
                print(f'Epoch {e+1}, Balance: {agent.portfolio_state[0,0]:.4f}')

            # next_data_window['date'] = pd.to_datetime(next_data_window['date'])
            # next_year = next_data_window['date'].iloc[0].year
            next_data_window = next_data_window.copy()
            next_data_window['date'] = pd.to_datetime(next_data_window['date'])
            next_years = next_data_window['date'].dt.strftime('%Y')
            next_year = next_years.iloc[0]

            if year != next_year:
                losses = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.55, 0.5, 0.35, 0.3, 0.25]
                for i in losses:
                    if agent.portfolio_state[0, 0] < INITIAL_AMOUNT * i:
                        distance_from_no_loss = 1 - i
                        final_penalty = -1000.0 * (distance_from_no_loss ** 2)

                profits = [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                           2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0]
                for i in profits:
                    if agent.portfolio_state[0, 0] > INITIAL_AMOUNT * i:
                        bonus = 1000.0 * (i**2)
                        final_penalty = bonus

                # keep track of yearly profit
                years_list_training.append(year)
                yearly_profit_training.append(agent.portfolio_state[0, 0])
                agent.reset_portfolio()

            if e==(agent.num_epochs-1):

                if year == final_training_year:
                    # track cumulated profits
                    cumulated_profit_per_epoch = agent.portfolio_state[0, 0]
                    cumulated_profits_list_training.append(cumulated_profit_per_epoch)
                    timestamps_list_training.append(agent.timestamp_portfolio)

            losses = [1,0.95,0.9,0.85,0.8,0.75,0.7,0.6,0.55,0.5,0.35,0.3,0.25]
            for i in losses:
                if agent.portfolio_state[0,0] < INITIAL_AMOUNT*i:
                    distance_from_no_loss = 1 - i
                    final_penalty = -1000.0 * (distance_from_no_loss ** 2)

            profits = [1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
            for i in profits:
                if agent.portfolio_state[0,0] > INITIAL_AMOUNT*i:
                    # print("Giving bonus")
                    bonus = 100.0 * (i**2)
                    final_penalty = bonus

            done = True if t == l_training - 1 else False

            if done:
                # keep track of yearly profit
                years_list_training.append(year)
                yearly_profit_training.append(agent.portfolio_state[0, 0])


        yearly_profit_training_avg = sum(yearly_profit_training) / len(yearly_profit_training)

        cumulated_profits_list_training_per_epcoh_list.append(yearly_profit_training_avg)
        epoch_numbers_history_training_for_profits.append(e+1)

        loss_per_epoch = sum(train_loss_history) / len(train_loss_history)
        print(f'Training Loss for Epoch {e+1}: {loss_per_epoch:.4f}')

        loss_history_per_epoch_training.append(loss_per_epoch)
        # epochs_list = [e+1]*len(train_loss_history)
        # epoch_numbers_history_training_loss.extend(epochs_list)
        epoch_numbers_history_training_loss.append(e+1)

        # printing portfolio state
        df_portfolio_state = pd.DataFrame(agent.portfolio_state, columns = cols_stocks)
        df_portfolio_state.insert(0, 'TIC', agent.portfolio_state_rows)
        print(f'Training: Portfolio state for epoch {e+1} is \n: {df_portfolio_state}')

        # Save explainability DataFrame for the last epoch
        if e == (agent.num_epochs-1):
            current_date = datetime.datetime.now()
            date_string = current_date.strftime("%Y-%m-%d_%H_%M")
            agent.explainability_df.to_csv(
                f'./reports/results_DQN/{reward_type}/{data_type}/training_last_epoch/training_explainability_{dataset_name}_{date_string}_{selected_data_entries}.csv', index=False)

        # VALIDATION PHASE ########################################################################
        agent.reset()
        agent.Q_network.eval()  # Set the model to evaluation mode
        agent.epsilon = 0.0 # no exploration
        unique_dates_validation = validation_data['date'].unique()
        max_profit = 0.0
        l_validation = len(unique_dates_validation)
        final_penalty = 1.0
        yearly_profit_validation = []
        years_list_validation = []

        for t in range(l_validation):

            ####
            data_window = validation_data.loc[(validation_data['date'] == unique_dates_validation[t])]

            # replace NaN values with 0.0
            data_window = data_window.fillna(0)
            dates = data_window['date'].tolist()

            state = getState(data_window, t, agent)

            # take action a, observe reward and next_state
            closing_prices = data_window['close'].tolist()

            action_index = agent.act_deterministic(state, closing_prices)

            # Find the indeces (stock_i, action_index_for_stock_i) where action_index  is present in action_index_arr_mask

            indices = np.where(action_index_arr_mask == action_index)
            stock_i, action_index_for_stock_i = map(int, indices)

            agent.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

            # Next state should append the t+1 data and portfolio_state. It also updates the position of agent portfolio based on agent position
            if t<(l_validation-1):
                next_data_window = validation_data.loc[(validation_data['date'] == unique_dates_validation[t + 1])]
                next_state = getState(next_data_window, t+1, agent)
                next_closing_prices = next_data_window['close'].tolist()
                next_dates = next_data_window['date'].tolist()
            else:
                next_state = state
                next_closing_prices = closing_prices
                next_dates = dates

            reward = agent.update_portfolio(next_closing_prices, next_dates, stock_i)
            agent.reward = reward*final_penalty

            state = torch.from_numpy(state).float().unsqueeze(0)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)

            with torch.no_grad():
                target_reward = (reward + agent.gamma * (
                    torch.max(agent.Q_network_val.forward(next_state), dim=1, keepdim=True)[0]).cpu().numpy() * (
                                              1 - done))[0]

            actions_indexes = torch.tensor([action_index], device=device)  # Create a tensor with a single index

            expected_rewards = agent.Q_network.forward(state)
            expected_reward = expected_rewards[0, action_index]  # Directly index the expected_rewards tensor

            target_reward_scalar = torch.tensor(target_reward, device=device).float()

            # Convert expected_reward to a float (assuming expected_reward is a scalar)
            expected_reward_float = expected_reward.item()

            loss = agent.loss_fn(torch.tensor(expected_reward_float, device=device).unsqueeze(0), target_reward_scalar)


            avg_loss = loss.mean().detach().cpu().numpy()
            agent.batch_loss_history.append(avg_loss)

            batch_loss_history = agent.batch_loss_history.copy()
            val_loss_history.extend(batch_loss_history)

            if (t%50 == 0 or t==l_validation-1) and len(val_loss_history)>0:
                loss_per_epoch_log = sum(val_loss_history) / len(val_loss_history)
                print(f'Validation Loss for Epoch {e+1}: {loss_per_epoch_log:.4f}')
                print(f'Epoch {e+1}, Balance: {agent.portfolio_state[0,0]:.4f}')

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

                # keep track of yearly profit
                yearly_profit_validation.append(agent.portfolio_state[0, 0])
                agent.reset_portfolio()

                # penalize agent
                losses = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.55, 0.5, 0.35, 0.3, 0.25]
                for i in losses:
                    if agent.portfolio_state[0, 0] < INITIAL_AMOUNT * i:
                        distance_from_no_loss = 1 - i
                        final_penalty = -1000.0 * (distance_from_no_loss ** 2)

                profits = [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                           2.0]
                for i in profits:
                    if agent.portfolio_state[0, 0] > INITIAL_AMOUNT * i:
                        bonus = 1000.0 * (i**2)
                        final_penalty = bonus

            # track profits only from the most recent year from last epoch:
            if e == (agent.num_epochs - 1):
                if year == final_validation_year:
                    cumulated_profit_per_epoch = agent.portfolio_state[0, 0]
                    cumulated_profits_list_validation.append(cumulated_profit_per_epoch)
                    timestamps_list_validation.append(agent.timestamp_portfolio)

            # penalyze agent
            losses = [1,0.95,0.9,0.85,0.8,0.75,0.7,0.6,0.55,0.5,0.35,0.3,0.25]
            for i in losses:
                if agent.portfolio_state[0,0] < INITIAL_AMOUNT*i:
                    distance_from_no_loss = 1 - i
                    final_penalty = -1000.0 * (distance_from_no_loss ** 2)

            profits = [1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
            for i in profits:
                if agent.portfolio_state[0,0] > INITIAL_AMOUNT*i:
                    # print("Giving bonus")
                    bonus = 100.0 * (i**2)
                    final_penalty = bonus

            done = True if t == l_validation - 1 else False
            if done:
                # keep track of yearly profit
                years_list_validation.append(year)
                yearly_profit_validation.append(agent.portfolio_state[0, 0])

        ####
        yearly_profit_validation_avg = sum(yearly_profit_validation) / len(yearly_profit_validation)

        cumulated_profits_list_validation_per_epcoh_list.append(yearly_profit_validation_avg)

        loss_per_epoch = (sum(val_loss_history) / len(val_loss_history))
        loss_history_per_epoch_validation.append(loss_per_epoch)
        epoch_numbers_history_validation_loss.append(e+1)
        epoch_numbers_history_val_for_profits.append(e+1)

        # loss_history_per_epoch_validation.append(val_loss_history)
        # epochs_list = [e+1]*len(val_loss_history)
        # epoch_numbers_history_validation_loss.extend(epochs_list)

        # printing portfolio state for validation at the end of the epoch run
        df_portfolio_state = pd.DataFrame(agent.portfolio_state, columns = cols_stocks)
        df_portfolio_state.insert(0, 'TIC', agent.portfolio_state_rows)
        print(f'Validation: Portfolio state for epoch {e+1} is \n: {df_portfolio_state}')

        # Save explainability DataFrame for the last epoch
        if e == (agent.num_epochs-1):
            current_date = datetime.datetime.now()
            date_string = current_date.strftime("%Y-%m-%d_%H_%M")
            agent.explainability_df.to_csv(
                f'./reports/results_DQN/{reward_type}/{data_type}/validation_last_epoch/validation_explainability_{dataset_name}_{date_string}_{selected_data_entries}.csv', index=False)


    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y-%m-%d_%H_%M")

    print(f'Debugging epoch_numbers_history_validation_loss is {epoch_numbers_history_validation_loss} \n loss_history_per_epoch_validation is {loss_history_per_epoch_validation}')

    # PLOTTING: Loss and Cumulated profits #######################################################
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 7))

    # Plot for training loss
    ax1.plot(epoch_numbers_history_training_loss, loss_history_per_epoch_training, label='Training Loss')
    ax1.plot(epoch_numbers_history_validation_loss, loss_history_per_epoch_validation, label='Validation Loss',
             color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Running Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.grid(True)  # Add a grid
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend()


    ax2.plot(epoch_numbers_history_training_for_profits, cumulated_profits_list_training_per_epcoh_list, label='Training Yearly Profit')
    ax2.plot(epoch_numbers_history_val_for_profits, cumulated_profits_list_validation_per_epcoh_list, label='Validation Yearly Profit',
             color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Yearly Profit')
    ax2.set_title('Yearly Profit per Epoch during \nTraining and Validation')
    ax2.grid(True)  # Add a grid
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend()

    # Plot for cumulative profits during validation
    ax3.plot(timestamps_list_training, cumulated_profits_list_training)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cumulated Profits')
    ax3.set_title(f'Cumulated Profits during Training - year {final_training_year}  \n(Last Epoch)')
    ax3.grid(True)
    ax3.legend()

    # Get the number of timestamps in the validation data
    num_timestamps = len(timestamps_list_training)

    # Calculate the step size for x-axis ticks
    step = max(1, num_timestamps // 6)  # Ensure at least 1 tick and round down

    # Set the x-axis tick locations and labels
    x_ticks = np.linspace(0, num_timestamps - 1, 6, dtype=int)
    x_tick_labels = [timestamps_list_training[i] for i in x_ticks]

    # Set the x-axis tick locations and labels
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_tick_labels, rotation=45)

    # Plot for cumulative profits during training
    ax4.plot(timestamps_list_validation, cumulated_profits_list_validation)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulated Profits')
    ax4.set_title(f'Cumulated Profits during Validation - year {final_validation_year} \n(Last Epoch)')
    ax4.grid(True)
    ax4.legend()
    # Get the number of timestamps in the validation data
    num_timestamps = len(timestamps_list_validation)
    # Calculate the step size for x-axis ticks
    step = max(1, num_timestamps // 6)  # Ensure at least 1 tick and round down
    # Set the x-axis tick locations and labels
    x_ticks = np.linspace(0, num_timestamps - 1, 6, dtype=int)
    x_tick_labels = [timestamps_list_validation[i] for i in x_ticks]

    # Set the x-axis tick locations and labels
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels(x_tick_labels, rotation=45)


    # Adjust layout and save the figure
    # Set the suptitle for the entire figure
    fig.suptitle('DQN RL Agent Training and Validation with Total Balance Return as reward')
    # Adjust the spacing at the top of the figure
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(f'./reports/figures/DQN_{reward_type}/{data_type}/DQN_training_loss_and_profits_for_{dataset_name}_{date_string}_{selected_data_entries}.png')
    plt.show()

    # save model
    torch.save(agent.Q_network.state_dict(), f'{output_filepath}/reward_return/trained_DQN-model_for_{dataset_name}_{date_string}_{selected_data_entries}.pth')
    torch.save(agent.Q_network_val.state_dict(), f'{output_filepath}/reward_return/trained_target-DQN-model_for_{dataset_name}_{date_string}_{selected_data_entries}.pth')

    # sva econfiguration file
    with open('./src/config_model_DQN_return.py', 'r') as file:
        config_contents = file.read()

    # Extract the relevant data from the config_contents string
    config_lines = [line.strip() for line in config_contents.split('\n') if '=' in line]
    config_data = '\n'.join(config_lines)
    config_text = config_data

    with open( f'{output_filepath}/reward_return/config_file_for_{dataset_name}_{date_string}_{selected_data_entries}.txt', 'w') as file:
        file.write(config_text)

    # TESTING ###############################################################################################################

    # Test Set No. 1
    print('Testing Phase')
    agent.reset()
    agent.epsilon = 0.0 # no exploration
    agent.memory = []
    cumulated_profits_list_testing = [INITIAL_AMOUNT]
    unique_dates_testing = test_data['date'].unique()
    dates_testing = [unique_dates_testing[0]]
    e = agent.num_epochs-1 #for explainability

    l_testing = len(unique_dates_testing)
    for t in range(l_testing):
        ####
        data_window = test_data.loc[(test_data['date'] == unique_dates_testing[t])]

        # replace NaN values with 0.0
        data_window = data_window.fillna(0)
        dates = data_window['date'].tolist()
        state = getState(data_window, t, agent)
        closing_prices = data_window['close'].tolist()
        # take action a, observe reward and next_state
        action_index = agent.act_deterministic(state, closing_prices)

        # Find the indeces (stock_i, action_index_for_stock_i) where action_index  is present in action_index_arr_mask
        indices = np.where(action_index_arr_mask == action_index)
        stock_i, action_index_for_stock_i = map(int, indices)

        agent.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

        reward = agent.update_portfolio(next_closing_prices, next_dates, stock_i)
        agent.reward = reward

        updated_balance = agent.portfolio_state[0, 0]
        cumulated_profits_list_testing.append(updated_balance)
        dates_testing.append(dates[0])

        if t < (l_testing - 1):
            next_data_window = test_data.loc[(test_data['date'] == unique_dates_testing[t + 1])]
            next_closing_prices = next_data_window['close'].tolist()
            next_dates = next_data_window['date'].tolist()
        else:
            next_state = state
            next_closing_prices = closing_prices
            next_dates = dates

        reward = agent.update_portfolio(next_closing_prices, next_dates, stock_i)

        # if agent.portfolio_state[0, 0] >= INITIAL_AMOUNT * 1.5:
        #     print("Have reached overshooting profits over 50%. Selling everything")
        #     agent.sell_everything(closing_prices, stock_i, h, e, dates)
        #     break

        state = next

        done = True if t == l_testing - 1 else False

    # printing portfolio state for testing at the end
    df_portfolio_state = pd.DataFrame(agent.portfolio_state, columns=cols_stocks)
    df_portfolio_state.insert(0, 'TIC', agent.portfolio_state_rows)
    print(f'Testing: Portfolio state is \n: {df_portfolio_state}')

    # saving the explainability file
    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y-%m-%d_%H_%M")
    agent.explainability_df.to_csv(
        f'./reports/results_DQN/{reward_type}/{data_type}/testing/testing_explainability_1_{dataset_name}_{date_string}_{selected_data_entries}.csv', index=False)


    # Test Set No. 2 ################################################################

    print('Testing Phase')
    agent.reset()
    agent.epsilon = 0.0  # no exploration
    agent.memory = []
    cumulated_profits_list_testing_2 = [INITIAL_AMOUNT]
    unique_dates_testing_2 = test_data_2['date'].unique()
    dates_testing_2 = [unique_dates_testing_2[0]]
    e = agent.num_epochs - 1  # for explainability

    l_testing_2 = len(unique_dates_testing_2)
    for t in range(l_testing_2):
        ####
        data_window = test_data_2.loc[(test_data_2['date'] == unique_dates_testing_2[t])]

        # replace NaN values with 0.0
        data_window = data_window.fillna(0)
        dates = data_window['date'].tolist()
        state = getState(data_window, t, agent)
        closing_prices = data_window['close'].tolist()
        # take action a, observe reward and next_state
        action_index = agent.act_deterministic(state, closing_prices)

        # Find the indeces (stock_i, action_index_for_stock_i) where action_index  is present in action_index_arr_mask
        indices = np.where(action_index_arr_mask == action_index)
        stock_i, action_index_for_stock_i = map(int, indices)

        agent.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

        reward = agent.update_portfolio(next_closing_prices, next_dates, stock_i)
        agent.reward = reward

        updated_balance = agent.portfolio_state[0, 0]
        cumulated_profits_list_testing_2.append(updated_balance)
        dates_testing_2.append(dates[0])

        if t < (l_testing_2 - 1):
            next_data_window = test_data_2.loc[(test_data_2['date'] == unique_dates_testing_2[t + 1])]
            next_closing_prices = next_data_window['close'].tolist()
            next_dates = next_data_window['date'].tolist()
        else:
            next_state = state
            next_closing_prices = closing_prices
            next_dates = dates

        reward = agent.update_portfolio(next_closing_prices, next_dates, stock_i)

        state = next

        done = True if t == l_testing_2 - 1 else False

    # printing portfolio state for testing at the end
    df_portfolio_state = pd.DataFrame(agent.portfolio_state, columns=cols_stocks)
    df_portfolio_state.insert(0, 'TIC', agent.portfolio_state_rows)
    print(f'Testing: Portfolio state is \n: {df_portfolio_state}')

    # saving the explainability file
    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y-%m-%d_%H_%M")
    agent.explainability_df.to_csv(
        f'./reports/results_DQN/{reward_type}/{data_type}/testing/testing_explainability_2_{dataset_name}_{date_string}_{selected_data_entries}.csv', index=False)



    # Plotting
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
    plt.margins(0.1)

    # Plot for cumulated_profits_list_testing on ax1
    ax1.plot(dates_testing, cumulated_profits_list_testing)
    ax1.set_title(f'Cumulated Profits Over Testing - year {final_testing_year}')
    ax1.set_xlabel("Dates")
    ax1.set_ylabel("Cumulated Profits")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)

    # Reduce the number of dates shown on the x-axis for ax1
    num_dates = 6
    skip = max(1, len(dates_testing) // num_dates)
    ax1.set_xticks(range(0, len(dates_testing), skip))
    ax1.set_xticklabels(dates_testing[::skip])

    # Plot for cumulated_profits_list_testing_2 on ax2
    ax2.plot(dates_testing_2, cumulated_profits_list_testing_2)
    ax2.set_title(f'Cumulated Profits during Testing - year {final_testing_year}')
    ax2.set_xlabel("Dates")
    ax2.set_ylabel("Cumulated Profits")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True)

    # Reduce the number of dates shown on the x-axis for ax2
    skip_2 = max(1, len(dates_testing_2) // num_dates)
    ax2.set_xticks(range(0, len(dates_testing_2), skip_2))
    ax2.set_xticklabels(dates_testing_2[::skip_2])

    # Adjust the bottom margin to make the x-axis labels visible
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Save the figure in the specified folder path
    plt.savefig(f'./reports/figures/DQN_{reward_type}/{data_type}/DQN_testing_profits_for_{dataset_name}_{date_string}_{selected_data_entries}.png')
    plt.show()

    # Saving results files

    # Losses per epoch
    loss_per_epoch_df = pd.DataFrame(
        {'epoch_no_training': epoch_numbers_history_training_loss,
         'loss_training': loss_history_per_epoch_training,
         'epoch_no_validation': epoch_numbers_history_validation_loss,
         'loss_validation': loss_history_per_epoch_validation
         })

    loss_per_epoch_df.to_csv(
        f'./reports/tables/results_DQN/{reward_type}/{data_type}/loss_per_epoch_{date_string}.csv', index=False)

    # Average yearly profit per epoch
    avg_yearly_profit_per_epoch = pd.DataFrame(
        {'epoch_no_training': epoch_numbers_history_training_for_profits,
         'avg_yearly_profit_training': cumulated_profits_list_training_per_epcoh_list,
         'epoch_no_validation': epoch_numbers_history_val_for_profits,
         'avg_yearly_profit_validation': cumulated_profits_list_validation_per_epcoh_list
         })

    avg_yearly_profit_per_epoch.to_csv(
        f'./reports/tables/results_DQN/{reward_type}/{data_type}/avg_yearly_profit_per_epoch_{dataset_name}_{date_string}.csv', index=False)

    # Profit per year - last epoch
    profit_per_year_last_epoch_training = pd.DataFrame(
        {'year_training': years_list_training,
         'profit_per_year_training': yearly_profit_training
         })

    profit_per_year_last_epoch_validation = pd.DataFrame(
        {
         'year_validation': years_list_validation,
         'profit_per_year_training': yearly_profit_validation
         })

    profit_per_year_last_epoch_training.to_csv(
        f'./reports/tables/results_DQN/{reward_type}/{data_type}/profit_per_year_last_epoch_training_{dataset_name}_{date_string}.csv', index=False)

    profit_per_year_last_epoch_validation.to_csv(
        f'./reports/tables/results_DQN/{reward_type}/{data_type}/profit_per_year_last_epoch_validation_{dataset_name}_{date_string}.csv', index=False)

    # Average profit for training (last epoch), validation (last_epoch) and testing
    avg_profit_training = sum(yearly_profit_training)/len(yearly_profit_training)
    avg_profit_validation = sum(yearly_profit_validation)/len(yearly_profit_validation)
    avg_profit_testing = cumulated_profits_list_testing[-1]

    avg_yearly_profit = [avg_profit_training, avg_profit_validation, avg_profit_testing]
    dataset_types = ['training_last_epoch','validation_last_epoch','testing_2022']

    avg_yearly_profit_df = pd.DataFrame(
        {'dataset': dataset_types,
         'avg_yearly_profit': avg_yearly_profit
         })

    avg_yearly_profit_df.to_csv(
        f'./reports/tables/results_DQN/{reward_type}/{data_type}/avg_yearly_profit_{dataset_name}_{date_string}.csv', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

