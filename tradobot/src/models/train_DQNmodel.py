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


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Trains the FinRL model.
    """
    logger = logging.getLogger(__name__)

    data = pd.read_csv(f'{input_filepath}/{DATASET}')[:201]
    print(f'Training on dataset: {input_filepath}/{DATASET}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Model running on {device}')
    h = hmax # user defined maximum amount to buy

    num_features = len(data.columns) + len(data.tic.unique()) - 1 #data_window after being preprocessed with one hot encoding

    agent = Agent(num_stocks=NUM_STOCKS, actions_dict=ACTIONS_DICTIONARY, h=h, num_features=num_features, balance=INITIAL_AMOUNT,
                  gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                  epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, tau=TAU, num_epochs=NUM_EPOCHS)

    action_index_arr_mask = np.arange(0, agent.num_actions * agent.num_stocks, 1, dtype=int).reshape(agent.num_stocks,
                                                                                                     agent.num_actions)
    unique_dates = data['date'].unique()
    cols_stocks = data['tic'].unique().tolist()

    l = len(unique_dates)-1

    loss_history = []
    loss_history_per_epoch = [] # will populate with: [(l1,epoch1),(l2,e1),..(l1,e_n),(l2,e_n)]
    epoch_numbers_history = []
    cumulated_profits_list = [INITIAL_AMOUNT]
    epoch_numbers_history_for_profits = [0]

    protfolio_state_rows = ['total_balance','position_per_stock','position_portfolio',
                            'daily_return_per_stock','daily_return_portfolio','cash_left',
                            'percentage_position_stock','shares_per_stock']

    for e in range(agent.num_epochs):
        print(f'Epoch {e}')

        agent.reset()

        date_index = unique_dates[0]
        data_window = data.loc[(data['date'] == unique_dates[0])]

        prev_closing_prices = data_window['close'].tolist()

        initial_state, agent = getState(data_window, 0, agent)

        # TODO: use this for the testing period. Update memory_step during the loop and add to an explainability DataFrame
        # initial_dates = getDates(data[0])
        # tickers = getTickers(data[0])
        # initial_closing_prices = getClosingPrices(data[0])
        #
        # initial_memory_step = np.stack([initial_dates, initial_closing_prices, agent.portfolio_state],axis=1)



        for t in range(l):

            date_index = unique_dates[t]
            data_window = data.loc[(data['date'] == unique_dates[t])]

            # replace NaN values with 0.0
            data_window = data_window.fillna(0)

            if t > 0:
                prev_closing_prices = data.loc[(data['date'] == unique_dates[t-1])]['close'].tolist()

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

            done = True if l==l-1 else False

            agent.remember(state=state, actions=(action_index_for_stock_i, stock_i, h), closing_prices=closing_prices,
                           reward=reward, next_state=next_state, done=done)

            state = next_state

            if len(agent.memory) >= agent.batch_size:
                agent.expReplay(e) # will also save the model on the last epoch

                batch_loss_history = agent.batch_loss_history.copy()
                loss_history.extend(batch_loss_history)


            if t%500 == 0 and len(loss_history)>0:
                loss_per_epoch_log = sum(loss_history) / len(loss_history)
                print(f'Training Loss for Epoch {e}: {loss_per_epoch_log:.4f}')

        current_date = datetime.datetime.now()

        #printing
        df_portfolio_state = pd.DataFrame(agent.portfolio_state, columns = cols_stocks)
        df_portfolio_state.insert(0, 'Info_row', protfolio_state_rows)
        #df_portfolio_state['info_row'] = protfolio_state_rows
        print(f'Portfolio state for epoch {e} is \n: {df_portfolio_state}')

        if len(loss_history)>0:
            # track loss per epoch
            loss_per_epoch = sum(loss_history) / len(loss_history)
            print(f'Training Loss for Epoch {e}: {loss_per_epoch:.4f}')
            loss_history_per_epoch.append(loss_per_epoch)
            epoch_numbers_history.append(e)
            epoch_numbers_history_for_profits.append(e)

            # track cumulated profits
            cumulated_profit_per_epoch = agent.portfolio_state[0,0]
            cumulated_profits_list.append(cumulated_profit_per_epoch)

        # Format the date as a readable string
        date_string = current_date.strftime("%d_%m_%Y")

    # Plot the running loss values after each epoch
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for training loss
    ax1.plot(epoch_numbers_history, loss_history_per_epoch)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Running Loss')
    ax1.set_title('Training Loss')

    # Plot for cumulative profits
    ax2.plot(epoch_numbers_history_for_profits, cumulated_profits_list)
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

