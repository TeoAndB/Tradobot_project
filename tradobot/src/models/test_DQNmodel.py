import datetime
# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.config_model_DQN import *
from src.models.DQN_model_fin import Agent, Portfolio, getState, maskActions, maskActions_evaluation
#from functions import *

import logging
import sys
import time
import click
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Trains the FinRL model.
    """
    logger = logging.getLogger(__name__)

    data = pd.read_csv(f'{input_filepath}/{DATASET}')[:927]
    print(f'Dataset used: {input_filepath}/{DATASET}')


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

    agent = Agent(num_stocks=NUM_STOCKS, actions_dict=ACTIONS_DICTIONARY, h=h, num_features=num_features, balance=INITIAL_AMOUNT, name_stocks=cols_stocks,
                  gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                  epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, tau=TAU, num_epochs=NUM_EPOCHS)

    action_index_arr_mask = np.arange(0, agent.num_actions * agent.num_stocks, 1, dtype=int).reshape(agent.num_stocks,
                                                                                                     agent.num_actions)

    # TRAINING #############################################################
    unique_dates_training = train_data['date'].unique()

    train_loss_history = []
    loss_history_per_epoch_training = [] # will populate with: [(l1,epoch1),(l2,e1),..(l1,e_n),(l2,e_n)]
    epoch_numbers_history_training = []
    cumulated_profits_list_training = [INITIAL_AMOUNT]
    epch_numbers_history_for_profits_training = [0]

    # VALIDATION  ###########################################################

    unique_dates_validation = validation_data['date'].unique()

    val_loss_history = []
    loss_history_per_epoch_validation = []
    epoch_numbers_history_validation = []
    cumulated_profits_list_validation = [INITIAL_AMOUNT]
    epoch_numbers_history_for_profits_validation = [0]


    for e in range(agent.num_epochs):
        print(f'Epoch {e}')

        agent.reset()

        data_window = train_data.loc[(train_data['date'] == unique_dates_training[0])]
        initial_state, agent = getState(data_window, 0, agent)

        # TRAINING PHASE ##################################################################

        l_training = len(unique_dates_training)

        for t in range(l_training):

            data_window = train_data.loc[(train_data['date'] == unique_dates_training[t])]

            # replace NaN values with 0.0
            data_window = data_window.fillna(0)
            dates = data_window['date'].tolist()

            state = getState(data_window, t, agent)

            # take action a, observe reward and next_state
            closing_prices = data_window['close'].tolist()

            action_index = agent.act(state, closing_prices)

            # Find the indeces (stock_i, action_index_for_stock_i) where action_index  is present in action_index_arr_mask

            indices = np.where(action_index_arr_mask == action_index)
            stock_i, action_index_for_stock_i = map(int, indices)

            reward = agent.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

            # Next state should append the t+1 data and portfolio_state. It also updates the position of agent portfolio based on agent position
            next_state, agent = getState(data_window, t+1, agent)

            done = True if t==l_training-1 else False


            agent.remember(state=state, actions=(action_index_for_stock_i, stock_i, h), closing_prices=closing_prices,
                           reward=reward, next_state=next_state, done=done)

            state = next_state

            if len(agent.memory) >= agent.batch_size:
                agent.expReplay(e) # will also save the model on the last epoch

                batch_loss_history = agent.batch_loss_history.copy()
                train_loss_history.extend(batch_loss_history)


            if t%500 == 0 and len(train_loss_history)>0:
                loss_per_epoch_log = sum(train_loss_history) / len(train_loss_history)
                print(f'Epoch {e}, Training Loss: {loss_per_epoch_log:.4f}')

        # printing portfolio state
        df_portfolio_state = pd.DataFrame(agent.portfolio_state, columns = cols_stocks)
        df_portfolio_state.insert(0, 'TIC', agent.portfolio_state_rows)
        print(f'Training: Portfolio state for epoch {e} is \n: {df_portfolio_state}')

        # keeping track of loss and cumulated profits
        if len(train_loss_history)>0:
            # track loss per epoch
            loss_per_epoch = sum(train_loss_history) / len(train_loss_history)
            print(f'Training Loss for Epoch {e}: {loss_per_epoch:.4f}')
            loss_history_per_epoch_training.append(loss_per_epoch)
            epoch_numbers_history_training.append(e)
            epch_numbers_history_for_profits_training.append(e)

            # track cumulated profits
            cumulated_profit_per_epoch = agent.portfolio_state[0,0]
            cumulated_profits_list_training.append(cumulated_profit_per_epoch)

        # Save explainability DataFrame for the last epoch
        if e == (agent.num_epochs-1):
            current_date = datetime.datetime.now()
            date_string = current_date.strftime("%Y-%m-%d")
            agent.explainability_df.to_csv(
                f'./reports/results_DQN/training_explainability_{DATASET}_{date_string}.csv', index=False)

        # VALIDATION PHASE ########################################################################
        agent.reset()
        agent.Q_network.eval()  # Set the model to evaluation mode
        val_loss_per_epoch = 0
        val_samples = 0
        agent.epsilon = 0.0 # no exploration
        unique_dates_validation = validation_data['date'].unique()

        l_validation = len(unique_dates_validation)
        for t in range(l_validation):

            ####
            data_window = validation_data.loc[(validation_data['date'] == unique_dates_validation[t])]

            # replace NaN values with 0.0
            data_window = data_window.fillna(0)
            dates = data_window['date'].tolist()

            state = getState(data_window, t, agent)

            # take action a, observe reward and next_state
            closing_prices = data_window['close'].tolist()

            action_index = agent.act(state, closing_prices)

            # Find the indeces (stock_i, action_index_for_stock_i) where action_index  is present in action_index_arr_mask

            indices = np.where(action_index_arr_mask == action_index)
            stock_i, action_index_for_stock_i = map(int, indices)

            reward = agent.execute_action(action_index_for_stock_i, closing_prices, stock_i, h, e, dates)

            # Next state should append the t+1 data and portfolio_state. It also updates the position of agent portfolio based on agent position
            next_state, agent = getState(data_window, t+1, agent)

            done = True if t==l_validation-1 else False


            agent.remember(state=state, actions=(action_index_for_stock_i, stock_i, h), closing_prices=closing_prices,
                           reward=reward, next_state=next_state, done=done)

            state = next_state

            if len(agent.memory) >= agent.batch_size:
                agent.expReplay_validation(e) # will also save the model on the last epoch

                batch_loss_history = agent.batch_loss_history.copy()
                val_loss_history.extend(batch_loss_history)

            ####

        # printing portfolio state for validation at the end of the epoch run
        df_portfolio_state = pd.DataFrame(agent.portfolio_state, columns = cols_stocks)
        df_portfolio_state.insert(0, 'TIC', agent.portfolio_state_rows)
        print(f'Validation: Portfolio state for epoch {e} is \n: {df_portfolio_state}')

        # keeping track of loss and cumulated profits
        if len(val_loss_history)>0:
            # track loss per epoch
            loss_per_epoch = sum(val_loss_history) / len(val_loss_history)
            print(f'Training Loss for Epoch {e}: {loss_per_epoch:.4f}')
            loss_history_per_epoch_validation.append(loss_per_epoch)
            epoch_numbers_history_validation.append(e)
            epoch_numbers_history_for_profits_validation.append(e)

            # track cumulated profits
            cumulated_profit_per_epoch = agent.portfolio_state[0,0]
            cumulated_profits_list_validation.append(cumulated_profit_per_epoch)


        # Save explainability DataFrame for the last epoch
        if e == (agent.num_epochs-1):
            current_date = datetime.datetime.now()
            date_string = current_date.strftime("%Y-%m-%d")
            agent.explainability_df.to_csv(
                f'./reports/results_DQN/validation_explainability_{DATASET}_{date_string}.csv', index=False)


    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%Y-%m-%d")


    # PLOTTING: Loss and Cumulated profits #######################################################
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for training loss
    ax1.plot(epoch_numbers_history_training, loss_history_per_epoch_training, label='Training Loss')
    ax1.plot(epoch_numbers_history_validation, loss_history_per_epoch_validation, label='Validation Loss',
             color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Running Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    # Plot for cumulative profits
    ax2.plot(epch_numbers_history_for_profits_training, cumulated_profits_list_training, label='Training Profits')
    ax2.plot(epoch_numbers_history_for_profits_validation, cumulated_profits_list_validation,
             label='Validation Profits', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cumulated Profits')
    ax2.set_title('Cumulated Profits during Training and Validation')
    ax2.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'./reports/figures/DQN_training_loss_and_profits_for_{DATASET}_{date_string}.png')
    plt.show()

    # save model
    torch.save(agent.Q_network.state_dict(), f'{output_filepath}/trained_DQN-model_for_{DATASET}_{date_string}.pth')
    torch.save(agent.Q_network_val.state_dict(), f'{output_filepath}/trained_target-DQN-model_for_{DATASET}_{date_string}.pth')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

