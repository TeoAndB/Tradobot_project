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

    data = pd.read_csv(f'{input_filepath}/{TRAIN_DATASET}')[:100]
    print(f'Training on dataset: {input_filepath}/{TRAIN_DATASET}')


    h = hmax # user defined maximum amount to buy

    num_features = len(data.columns) + len(data.tic.unique()) - 1 #data_window after being preprocessed with one hot encoding

    agent = Agent(num_stocks=NUM_STOCKS, actions_dict=ACTIONS_DICTIONARY, num_features=num_features, balance=INITIAL_AMOUNT,
                  gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN,
                  epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)

    l = len(data)-1

    loss_history = [] # will populate with: [(l1,epoch1),(l2,e1),..(l1,e_n),(l2,e_n)]
    epoch_numbers_history = []

    for e in range(NUM_EPOCHS):
        print(f'Epoch {e}')

        agent.reset()

        unique_dates = data['date'].unique()

        date_index = unique_dates[0]
        data_window = data.loc[(data['date'] == unique_dates[0])]

        prev_closing_prices = data_window['close'].tolist()

        initial_state, agent = getState(data_window, 0, prev_closing_prices, agent)

        # TODO: use this for the testing period. Update memory_step during the loop and add to an explainability DataFrame
        # initial_dates = getDates(data[0])
        # tickers = getTickers(data[0])
        # initial_closing_prices = getClosingPrices(data[0])
        #
        # initial_memory_step = np.stack([initial_dates, initial_closing_prices, agent.portfolio_state],axis=1)



        for t in range(l):

            date_index = unique_dates[t]
            data_window = data.loc[(data['date'] == unique_dates[t])]
            if t > 0:
                prev_closing_prices = data.loc[(data['date'] == unique_dates[t-1])]['close'].tolist()

            state = getState(data_window, t, prev_closing_prices, agent)

            action_index = agent.act(state)

            for i in range(agent.num_stocks):
                if action_index > i * agent.num_actions: #action space for a single stock
                    action_index_for_stock_i = action_index - i * agent.num_stocks
                else:
                    action_index_for_stock_i = action_index

            stock_i = int(action_index // agent.num_actions) #find which stock the actoon is performed for

            # take action a, observe reward and next_state
            close_price_stock_i = data_window.loc[stock_i,'close']
            reward = agent.execute_action(action_index_for_stock_i, close_price_stock_i, stock_i, h)

            # add close prices, tickers and agent's memory step to explainability DataFrame

            # Next state should append the t+1 data and portfolio_state. It also updates the position of agent portfolio based on agent position
            next_state, agent = getState(data, t+1, prev_closing_prices, agent)

            done = True if l==l-1 else False
            agent.remember(state=state, actions=(action_index_for_stock_i,stock_i, h), reward=reward, next_state=next_state, done=done)

            state = next_state

            if len(agent.memory) > agent.batch_size:
                agent.expReplay(e) # will also save the model on the last epoch
                loss_history.extend(agent.batch_loss_history)
                epoch_numbers_history.extent(agent.epoch_numbers)

        print(loss_history)

        current_date = datetime.now()

        # Format the date as a readable string
        date_string = current_date.strftime("%d_%m_%Y")

        # Plot the running loss values after each epoch
        plt.plot(epoch_numbers_history, loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Running Loss')
        plt.title('Training Loss')
        plt.savefig(f'./reports/figures/DQN_training_loss_plot_{date_string}.png')
        plt.show()

    # TODO: save trained model in output_filepath

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

