import datetime
# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.config_model_DQN import *
from src.models.DQN_model_fin import Agent, Portfolio, getState
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

    data = pd.read_csv(f'{input_filepath}/{DATASET}')[:21708]
    data['date'] = pd.to_datetime(data['date'])

    # Set the seed for reproducibility
    random_seed = 42
    # using unique dates for splitting (not timestamps)
    unique_dates_days = pd.to_datetime(data['date']).dt.date.unique().tolist()

    print(f'Len of unique_dates_days is {len(unique_dates_days)}')

    # Split unique_dates_days into train, validation, and test sets
    train_dates, remaining_dates = train_test_split(unique_dates_days, test_size=0.25, random_state=random_seed)
    validation_dates, test_dates = train_test_split(remaining_dates, test_size=0.4, random_state=random_seed)

    # Create the train, validation, and test DataFrames based on the selected dates
    train_data = data[data['date'].dt.date.isin(train_dates)]
    validation_data = data[data['date'].dt.date.isin(validation_dates)]
    test_data = data[data['date'].dt.date.isin(test_dates)]


    # for explainability
    protfolio_state_rows = ['total_balance','position_per_stock','position_portfolio',
                            'daily_return_per_stock','daily_return_portfolio','cash_left',
                            'percentage_position_stock','shares_per_stock']


    print(f'Training on dataset: {input_filepath}/{DATASET}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model running on {device}')
    h = hmax # user defined maximum amount to buy

    num_features = len(data.columns) + len(data.tic.unique()) - 1 #data_window after being preprocessed with one hot encoding

    cols_stocks = data['tic'].unique().tolist()

    agent = Agent(num_stocks=NUM_STOCKS, actions_dict=ACTIONS_DICTIONARY, h=h, num_features=num_features, balance=INITIAL_AMOUNT,
                  gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                  epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, tau=TAU, num_epochs=NUM_EPOCHS)

    action_index_arr_mask = np.arange(0, agent.num_actions * agent.num_stocks, 1, dtype=int).reshape(agent.num_stocks,
                                                                                                     agent.num_actions)

    # TRAINING #############################################################################
    unique_dates_training = train_data['date'].unique()

    train_loss_history = []
    loss_hsitory_per_epoch_training = [] # will populate with: [(l1,epoch1),(l2,e1),..(l1,e_n),(l2,e_n)]
    epoch_numbers_history_training = []
    cumulates_profits_list_training = [INITIAL_AMOUNT]
    epch_numbers_history_for_profits_training = [0]


    for e in range(agent.num_epochs):
        print(f'Epoch {e}')

        agent.reset()

        data_window = train_data.loc[(train_data['date'] == unique_dates_training[0])]

        prev_closing_prices = data_window['close'].tolist()

        initial_state, agent = getState(data_window, 0, agent)

        # TODO: use this for the testing period. Update memory_step during the loop and add to an explainability DataFrame
        # initial_dates = getDates(data[0])
        # tickers = getTickers(data[0])
        # initial_closing_prices = getClosingPrices(data[0])
        #
        # initial_memory_step = np.stack([initial_dates, initial_closing_prices, agent.portfolio_state],axis=1)

        t = 0
        l_training = len(unique_dates_training)

        for t in range(l_training):
            curr_date_day = pd.to_datetime(unique_dates_training[t]).date()

            data_window = train_data.loc[(train_data['date'] == unique_dates_training[t])]

            # replace NaN values with 0.0
            data_window = data_window.fillna(0)

            #print(f'data_window is {data_window}')

            if t > 0:
                prev_date_day = pd.to_datetime(unique_dates_training[t-1]).date()
                prev_closing_prices = train_data.loc[(data['date'] == unique_dates_training[t-1])]['close'].tolist()

            state = getState(data_window, t, agent)

            # take action a, observe reward and next_state
            closing_prices = data_window['close'].tolist()

            action_index = agent.act(state, closing_prices)

            # Find the indeces (stock_i, action_index_for_stock_i) where action_index  is present in action_index_arr_mask

            indices = np.where(action_index_arr_mask == action_index)
            stock_i, action_index_for_stock_i = map(int, indices)

            reward = agent.execute_action(action_index_for_stock_i, closing_prices, stock_i, h)

            # add close prices, tickers and agent's memory step to explainability DataFrame

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

        current_date = datetime.datetime.now()

        #printing
        df_portfolio_state = pd.DataFrame(agent.portfolio_state, columns = cols_stocks)
        df_portfolio_state.insert(0, 'Info_row', protfolio_state_rows)
        #df_portfolio_state['info_row'] = protfolio_state_rows
        print(f'Portfolio state for epoch {e} is \n: {df_portfolio_state}')

        if len(train_loss_history)>0:
            # track loss per epoch
            loss_per_epoch = sum(train_loss_history) / len(train_loss_history)
            print(f'Training Loss for Epoch {e}: {loss_per_epoch:.4f}')
            loss_hsitory_per_epoch_training.append(loss_per_epoch)
            epoch_numbers_history_training.append(e)
            epch_numbers_history_for_profits_training.append(e)

            # track cumulated profits
            cumulated_profit_per_epoch = agent.portfolio_state[0,0]
            cumulates_profits_list_training.append(cumulated_profit_per_epoch)

        # Format the date as a readable string
        date_string = current_date.strftime("%d_%m_%Y")

    # Plot the running loss values after each epoch
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for training loss
    ax1.plot(epoch_numbers_history_training, loss_hsitory_per_epoch_training)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Running Loss')
    ax1.set_title('Training Loss')

    # Plot for cumulative profits
    ax2.plot(epch_numbers_history_for_profits_training, cumulates_profits_list_training)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cumulated Profits')
    ax2.set_title('Cumulated Profits during Training')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'./reports/figures/DQN_training_loss_and_profits_for_{DATASET}_{date_string}.png')
    plt.show()

    torch.save(agent.Q_network.state_dict(), f'{output_filepath}/trained_DQN-model_for_{DATASET}_{date_string}.pth')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

