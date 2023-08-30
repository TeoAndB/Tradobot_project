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

    selected_adjustments = 'regular_DQN_100Epochs'
    #data_type = "minute_frequency_data"
    data_type = "daily_frequency_data"
    reward_type = "reward_portfolio_return"

    print(f'Dataset used: {input_filepath}/{DATASET}')

    dataset_name = os.path.splitext(DATASET)[0]


    # Create an empty DataFrames for explainability
    cols_stocks = data['tic'].unique().tolist()

    # Set the seed for reproducibility
    unique_dates = data['date'].unique().tolist()

    # Define the desired date ranges
    train_start_date = '2014-06-01'
    train_end_date = '2019-12-31'
    final_training_year = '2019'

    # validation
    val_start_date = '2020-01-01'
    val_end_date = '2022-12-31'
    final_validation_year = '2022'

    # testing
    test_start_date = '2021-01-01'
    test_end_date = '2021-12-31'
    final_testing_year = '2021'

    # extra insights for year 2021
    test_start_date_2 = '2022-01-01'
    test_end_date_2 = '2022-12-31'
    final_testing_year_2 = '2022'

    # Filter data based on date ranges
    train_data = data[(data['date'] >= train_start_date) & (data['date'] <= train_end_date)]
    validation_data = data[(data['date'] >= val_start_date) & (data['date'] <= val_end_date)]
    test_data = data[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]
    test_data_2 = data[(data['date'] >= test_start_date_2) & (data['date'] <= test_end_date_2)]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model running on {device}')
    h = hmax # user defined maximum amount to buy

    num_features = len(data.columns) + len(data.tic.unique()) - 1 #data_window after being preprocessed with one hot encoding

    models_folder = './models/reward_return/'
    name_DQN = f'{models_folder}trained_DQN-model_for_dataset1_1Day_w14Lags_HA-WBA-INCY_2023-08-23_20_27_with_CQL_100_Epochs.pth'
    name_DQN_target = f'{models_folder}trained_target-DQN-model_for_dataset1_1Day_w14Lags_HA-WBA-INCY_2023-08-23_20_27_with_CQL_100_Epochs.pth'


    agent = Agent(num_stocks=NUM_STOCKS, actions_dict=ACTIONS_DICTIONARY, h=h, num_features=num_features, balance=INITIAL_AMOUNT, name_stocks=cols_stocks,
                  gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                  epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, tau=TAU, num_epochs=NUM_EPOCHS,
                  model_path=name_DQN,
                  model_target_path=name_DQN_target
                  )

    action_index_arr_mask = np.arange(0, agent.num_actions * agent.num_stocks, 1, dtype=int).reshape(agent.num_stocks,
                                                                                                     agent.num_actions)

    # SPLITTING DATA INTO EPISODES FOR TRAINING AND VALIDATION ##################################################################

    # Partition the unique dates by year
    train_data = train_data.copy()
    train_data['date'] = pd.to_datetime(train_data['date'])
    unique_dates_training = pd.to_datetime(train_data['date'].unique())

    dates_list_by_year = {}
    for date in unique_dates_training:
        year = date.year
        if year not in dates_list_by_year:
            dates_list_by_year[year] = []
        dates_list_by_year[year].append(date)

    validation_data = validation_data.copy()
    validation_data['date'] = pd.to_datetime(validation_data['date'])

    unique_dates_validation = validation_data['date'].unique()

    dates_list_by_year_val = {}
    for date in unique_dates_validation:
        year = pd.Timestamp(date).year
        if year not in dates_list_by_year_val:
            dates_list_by_year_val[year] = []
        dates_list_by_year_val[year].append(date)

    # VALIDATION lists for keeping track of losses and profits ###########################################################

    validation_loss_history = []
    loss_history_per_epoch_validation = []
    epoch_numbers_history_validation = []
    cumulated_profits_list_validation_per_epcoh_list = [INITIAL_AMOUNT]
    epoch_numbers_history_val_for_profits = [0]
    epoch_numbers_history_validation_loss = []
    utility_list_validation_per_epoch = [0.0]

    timestamps_list_validation = [f'{final_validation_year}-01-01']


        # VALIDATION PHASE ########################################################################
        agent.epsilon = 0.0 # no exploration

        # final_penalty = 1.0
        yearly_balance_validation = []
        validation_data = validation_data.copy()
        yearly_balance_validation_last_epoch = []
        years_list_validation_last_epoch = []

        for year, year_episode_dates in dates_list_by_year_val.items():
            print(f'Epoch {e+1}, year {year}')
            agent.soft_reset()
            agent.reset_portfolio()

            data_window = validation_data.loc[(validation_data['date'] == year_episode_dates[0])]
            initial_state = getState(data_window, 0, agent)
            closing_prices = data_window['close'].tolist()

            l_validation = len(year_episode_dates)
            max_profit = 0.0
            loss_per_50_timesteps = []
            yearly_utility_list_validation = []

            cumulated_rewards_list_validation = [0.0]
            cumulated_profits_list_validation = [INITIAL_AMOUNT]

            for t in range(l_validation):

                data_window = validation_data.loc[(validation_data['date'] == year_episode_dates[t])]

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

                # act epsilon-greedy
                action_index = agent.act_deterministic(state, closing_prices)

                indices = np.where(action_index_arr_mask == action_index)
                stock_i, action_index_for_stock_i = map(int, indices)

                # Execute action
                amount_transaction = agent.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

                # Next state should append the t+1 data and portfolio_state. It also updates the position of agent portfolio based on agent position

                if t < (l_validation - 1):
                    next_data_window = validation_data.loc[(validation_data['date'] == year_episode_dates[t + 1])]
                    next_state = getState(next_data_window, t + 1, agent)
                    next_closing_prices = next_data_window['close'].tolist()
                    next_dates = next_data_window['date'].tolist()
                else:
                    next_state = state
                    next_closing_prices = closing_prices
                    next_dates = dates

                # Update portfolio and observe reward
                reward = agent.update_portfolio(next_closing_prices, next_dates, stock_i, amount_transaction)

                done = True if t == l_validation - 1 else False

                agent.remember(state=state, actions=(action_index, action_index_for_stock_i, stock_i, h),
                               closing_prices=closing_prices,
                               reward=reward, next_state=next_state, done=done)

                if len(agent.memory) >= agent.batch_size:
                    for i in range(NUM_SAMPLING):
                        agent.expReplay(e)

                        batch_loss_history = agent.batch_loss_history.copy()
                        validation_loss_history.extend(batch_loss_history)

                if (t % 50 == 0 or t == (l_validation - 1)) and len(validation_loss_history) > 0:
                    loss_per_epoch_log = sum(validation_loss_history) / len(validation_loss_history)
                    loss_per_50_timesteps.append(loss_per_epoch_log)
                    print(f'Epoch {e + 1}, year {year}, Validation Loss: {loss_per_epoch_log:.4f}')
                    print(f'Epoch {e + 1}, year {year}, Balance: {agent.portfolio_state[0, 0]:.4f}')
                    print(f'Epoch {e + 1}, year {year}, Utility: {agent.utility:.4f}')

                next_data_window = next_data_window.copy()
                next_data_window['date'] = pd.to_datetime(next_data_window['date'])
                next_years = next_data_window['date'].dt.strftime('%Y')

                # track profits for the final year during the last epoch
                if year == final_validation_year and e == NUM_EPOCHS - 1:
                    cumulated_profit_per_epoch = agent.portfolio_state[0, 0]
                    cumulated_profits_list_validation.append(cumulated_profit_per_epoch)
                    cumulated_rewards_list_validation.append(agent.utility)
                    timestamps_list_validation.append(agent.timestamp_portfolio.date().strftime('%Y-%m-%d'))

                done = True if t == l_validation - 1 else False

                if done and e == NUM_EPOCHS - 1:
                    years_list_validation_last_epoch.append(year)
                    yearly_balance_validation_last_epoch.append(agent.portfolio_state[0, 0])

                if done:
                    # keep track of yearly profit
                    yearly_balance_validation.append(agent.portfolio_state[0, 0])
                    yearly_utility_list_validation.append(agent.utility)

        ####
        average_utility_validation = sum(yearly_utility_list_validation)/len(yearly_utility_list_validation)
        yearly_balance_validation_avg = sum(yearly_balance_validation) / len(yearly_balance_validation)

        cumulated_profits_list_validation_per_epcoh_list.append(yearly_balance_validation_avg)
        utility_list_validation_per_epoch.append(average_utility_validation)


        loss_per_epoch = (sum(validation_loss_history) / len(validation_loss_history))
        loss_history_per_epoch_validation.append(loss_per_epoch)
        epoch_numbers_history_validation_loss.append(e+1)
        epoch_numbers_history_val_for_profits.append(e+1)


        # printing portfolio state for validation at the end of the epoch run
        df_portfolio_state = pd.DataFrame(agent.portfolio_state, columns = cols_stocks)
        df_portfolio_state.insert(0, 'TIC', agent.portfolio_state_rows)
        print(f'Validation: Portfolio state for epoch {e+1} is \n: {df_portfolio_state}')

        # Save explainability DataFrame for the last epoch
        if e == (agent.num_epochs-1):
            current_date = datetime.datetime.now()
            date_string = current_date.strftime("%Y-%m-%d_%H_%M")
            agent.explainability_df.to_csv(
                f'./reports/results_DQN/{reward_type}/{data_type}/validation_last_epoch/validation_explainability_{dataset_name}_{date_string}_{selected_adjustments}.csv', index=False)

    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y-%m-%d_%H_%M")


    # PLOTTING: Loss, Utility and Cumulated profits #######################################################
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

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

    ax2.plot(epoch_numbers_history_training_for_profits, utility_list_training_per_epoch, label='Training Avg. Yearly Utility')
    ax2.plot(epoch_numbers_history_val_for_profits, utility_list_validation_per_epoch, label='Validation Avg. Yearly Utiliy',
             color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Yearly Utility')
    ax2.set_title('Average Yearly Utility per Epoch during \nTraining and Validation')
    ax2.grid(True)  # Add a grid
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend()

    ax3.plot(epoch_numbers_history_training_for_profits, cumulated_profits_list_training_per_epcoh_list, label='Training Avg. Yearly Balance')
    ax3.plot(epoch_numbers_history_val_for_profits, cumulated_profits_list_validation_per_epcoh_list, label='Validation Avg. Yearly Balance',
             color='orange')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Yearly Balance')
    ax3.set_title('Average Yearly Balance per Epoch during \nTraining and Validation')
    ax3.grid(True)  # Add a grid
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.legend()

    # Adjust the spacing at the top of the figure
    fig.suptitle('DQN Average Loss, Average Utility and Average Balance')
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(f'./reports/figures/DQN_{reward_type}/{data_type}/DQN_trainingLoss_Utility_Profits_{dataset_name}_{date_string}_{selected_adjustments}.png')
    # plt.show()

     # PLOTTING: Cumulated profits  and Utilities for the last training/validation year #######################################################
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 7))

    ax1.plot(timestamps_list_training, cumulated_profits_list_training)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Monetary Balance')
    ax1.set_title(f'Monetary Balance during Training - year {final_training_year}  \n(Last Epoch)')
    ax1.grid(True)
    ax1.legend()


    ax2.plot(timestamps_list_training, cumulated_rewards_list_training)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulated Rewards (Utility)')
    ax2.set_title(f'Cumulated Rewards (Utility) during Training - year {final_training_year}  \n(Last Epoch)')
    ax2.grid(True)
    ax2.legend()

    # Get the number of timestamps in the validation data
    num_timestamps = len(timestamps_list_training)

    # Calculate the step size for x-axis ticks
    step = max(1, num_timestamps // 6)  # Ensure at least 1 tick and round down

    # Set the x-axis tick locations and labels
    x_ticks = np.linspace(0, num_timestamps - 1, 6, dtype=int)
    x_tick_labels = [timestamps_list_training[i] for i in x_ticks]

    # Set the x-axis tick locations and labels
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels, rotation=45)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels, rotation=45)

    ax3.plot(timestamps_list_validation, cumulated_profits_list_validation)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Monetary Balance')
    ax3.set_title(f'Monetary Balance during Validation - year {final_validation_year} \n(Last Epoch)')
    ax3.grid(True)
    ax3.legend()

    ax4.plot(timestamps_list_validation, cumulated_rewards_list_validation)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulated Rewards (Utility)')
    ax4.set_title(f'Cumulated Rewards (Utility) during Validation - year {final_validation_year} \n(Last Epoch)')
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
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_tick_labels, rotation=45)
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels(x_tick_labels, rotation=45)


    # Adjust layout and save the figure
    # Set the suptitle for the entire figure
    fig.suptitle('DQN Utility and Monetary Balance')
    # Adjust the spacing at the top of the figure
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(f'./reports/figures/DQN_{reward_type}/{data_type}/DQN_Utility_Porifts_lastYear_{dataset_name}_{date_string}_{selected_adjustments}.png')
    # plt.show()

    # save model
    torch.save(agent.Q_network.state_dict(), f'{output_filepath}/reward_return/trained_DQN-model_for_{dataset_name}_{date_string}_{selected_adjustments}.pth')
    torch.save(agent.Q_network_val.state_dict(), f'{output_filepath}/reward_return/trained_target-DQN-model_for_{dataset_name}_{date_string}_{selected_adjustments}.pth')

    # sva econfiguration file
    with open('./src/config_model_DQN_return.py', 'r') as file:
        config_contents = file.read()

    # Extract the relevant data from the config_contents string
    config_lines = [line.strip() for line in config_contents.split('\n') if '=' in line]
    config_data = '\n'.join(config_lines)
    config_text = config_data

    with open( f'{output_filepath}/reward_return/config_file_for_{dataset_name}_{date_string}_{selected_adjustments}.txt', 'w') as file:
        file.write(config_text)

    # Saving results_FinRL files

    # Losses per epoch
    loss_per_epoch_df = pd.DataFrame(
        {'epoch_no_training': epoch_numbers_history_training_loss,
         'loss_training': loss_history_per_epoch_training,
         'epoch_no_validation': epoch_numbers_history_validation_loss,
         'loss_validation': loss_history_per_epoch_validation
         })

    loss_per_epoch_df.to_csv(
        f'./reports/tables/results_DQN/{reward_type}/{data_type}/loss_per_epoch_{date_string}_{selected_adjustments}.csv', index=False)


    # Average yearly profit per epoch
    avg_yearly_balance_per_epoch = pd.DataFrame(
        {'epoch_no': epoch_numbers_history_training_for_profits,
         'avg_yearly_balance_training': cumulated_profits_list_training_per_epcoh_list,
         'avg_yearly_proft_training': [balance / INITIAL_AMOUNT for balance in cumulated_profits_list_training_per_epcoh_list],
         'avg_yearly_balance_validation': cumulated_profits_list_validation_per_epcoh_list,
         'avg_yearly_proft_validation': [balance / INITIAL_AMOUNT for balance in
                                       cumulated_profits_list_validation_per_epcoh_list]
         })

    avg_yearly_balance_per_epoch.to_csv(
        f'./reports/tables/results_DQN/{reward_type}/{data_type}/avg_yearly_balance_per_epoch_{dataset_name}_{date_string}_{selected_adjustments}.csv', index=False)

    print(f'DQN Model: Overall Average yearly profit \n{avg_yearly_balance_per_epoch}')

    # Profit per year - last epoch
    profit_per_year_last_epoch_training = pd.DataFrame(
        {'year_training': years_list_training,
         'balance_per_year_training': yearly_balance_training_last_epoch,
         'profit_per_year_training': [balance / INITIAL_AMOUNT for balance in yearly_balance_training_last_epoch]
         })


    profit_per_year_last_epoch_validation = pd.DataFrame(
        {
         'year_validation': years_list_validation_last_epoch,
         'balance_per_year_validation': yearly_balance_validation_last_epoch,
         'profit_per_year_validation': [balance / INITIAL_AMOUNT for balance in yearly_balance_validation_last_epoch]
         })

    profit_per_year_last_epoch_training.to_csv(
        f'./reports/tables/results_DQN/{reward_type}/{data_type}/profit_per_year_last_epoch_training_{dataset_name}_{date_string}_{selected_adjustments}.csv', index=False)

    print(f'DQN Model: Profit per Year for Last Epoch - Training \n{profit_per_year_last_epoch_training}')


    profit_per_year_last_epoch_validation.to_csv(
        f'./reports/tables/results_DQN/{reward_type}/{data_type}/profit_per_year_last_epoch_validation_{dataset_name}_{date_string}_{selected_adjustments}.csv', index=False)

    print(f'DQN Model: Profit per Year for Last Epoch - Validation \n{profit_per_year_last_epoch_validation}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

