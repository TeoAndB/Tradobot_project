import argparse
import importlib
import logging
import sys
import time

import click
import numpy as np
# np.random.seed(3)  # for reproducible Keras operations

import datetime
import pandas as pd

from src.config_model import *
from src.models.DQN_model import Portfolio,DQN_Agent


# immutable variable
from src.models.DQN_model2 import Agent



#TODO: write all functions
def hold(self):
    logging.info('Hold')

# def buy(stock_index, amount):
#     agent.balance -= stock_prices[t]
#     agent.inventory.append(stock_prices[t])
#     agent.buy_dates.append(t)
#     logging.info('Buy:  ${:.2f}'.format(stock_prices[t]))

def buy_1share(self, stock_index):
    # somehow make the agent interract?
    # update portfolio

def buy_1share(self, stock_index):

def sell_t(self, stock_index):
    t = self.t








# TODO: INSERT TRAINING LOOP HERE:
# act output is of type: (action_stock1, action_stock2, ...)
# Q_target_values per stock [Q_val1, Q_val2, ...]
# backpropagation happens based on MSE, loss according to all actions?

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Trains the FinRL model.
    """
    logger = logging.getLogger(__name__)

    df = pd.read_csv(f'{input_filepath}/{TRAIN_DATASET}')
    num_stocks = len(df.tic.unique())

    stock_name = TRAIN_DATASET
    window_size = rebalance_window




    initial_balance = initial_amount

    agent = Agent(state_dim=num_stocks, balance=initial_balance)

    # dictionary for each action index
    action_dict = {0: 'Hold', 2: 'Buy_1share', 3: 'Buy_24p', 4: 'Buy_50p', 5: 'Buy_100p',
                   6: 'Sell_1share', 7: 'Sell_25p', 8: 'Sell_50p', 9: 'Sell_75p', 10: 'Sell100p',
                   11: 'Buy_t', 12: 'Sell_t'  # user defined
                   }

    # actions switcher
    switch = {
        'Hold': hold,

        # Buy % of stock value already in the portfolio
        'Buy_1share': buy_1share,
        'Buy_24p': buy_24p,
        'Buy_50p': buy_50p,
        'Buy_75p': buy_75p,
        'Buy_100p': buy_100p,
        # buy user defined cash amount
        'Buy_t': buy_t,

        # Sell % of stock value already in the portfolio
        'Sell_1share': sell_1share,
        'Sell_24p': sell_24p,
        'Sell_50p': sell_50p,
        'Sell_75p': sell_75p,
        'Sell_100p': sell_100p,
        # buy user defined cash amount
        'Sell_t': sell_t
    }

    # calculate episode numbers
    date_format = "%Y/%m/%d"
    a = datetime.strptime(TRAIN_START_DATE, date_format)
    b = datetime.strptime(TRAIN_END_DATE, date_format)
    delta = b - a
    trading_period = delta.days
    num_episode = int(np.floor(trading_period/window_size))

    logging.info(f'Trading Object:           {stock_name}')
    logging.info(f'Training Period:           {TRAIN_START_DATE} - {TRAIN_END_DATE}')
    logging.info(f'Rebalance Window Size:              {window_size} days')
    logging.info(f'Training Episode:         {num_episode}')
    logging.info(f'Model Name:               DQN')
    logging.info('Initial Portfolio Value: ${:,}'.format(initial_balance))



    start_time = time.time()
    for e in range(1, num_episode + 1):
        logging.info(f'\nEpisode: {e}/{num_episode}')

        agent.reset()  # reset to initial balance and epsilon

        # loop through window_size of days- for each day date make a state...


        # USE switch[action_name](stock_index)




# [date_1, row1...]
# [date_2, row2..]
#num stocks