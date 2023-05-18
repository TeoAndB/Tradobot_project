import argparse
import importlib
import logging
import sys

import numpy as np
# np.random.seed(3)  # for reproducible Keras operations

from utils import *
from collections import MappingProxyType
import pandas as pd

#
# parser = argparse.ArgumentParser(description='command line options')
# parser.add_argument('--model_to_load', action="store", dest="model_to_load", default='DQN_ep10', help="model name")
# parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2018', help="stock name")
# parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')
# inputs = parser.parse_args()
from src.models.DQN_model import Portfolio,DQN_Agent

model_to_load = inputs.model_to_load
model_name = model_to_load.split('_')[0]
stock_name = inputs.stock_name
initial_balance = inputs.initial_balance
display = True
window_size = 10


# immutable variable
action_dict = {0: 'Hold', 2: 'Buy_1share', 3:'Buy_24p', 4:'Buy_50p', 5:'Buy_100p',
               6:'Sell_1share', 7:'Sell_25p',8:'Sell_50p', 9:'Sell_75p', 10:'Sell100p',
               11: 'Buy_t', 12: 'Sell_t' #user defined
}





# Create an immutable dictionary
immutable_dict = MappingProxyType(action_dict)



agent =DQN_Agent(portfolio)

class DQN_Environment():
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, action_dict,
        df: pd.DataFrame,
        stock_dim: int,
        results: list,
        hmax: int,
        NN_output: list,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        state: np.array,
        switch: dict,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",

        t = 100 #user defined cash amount, eg. 100$,


     ):
        self.df = df
        self.stock_dim = stock_dim
        self.state = self._initiate_state()

        self.results = results
        self.NN_output = NN_output #[(action, stock1),(action. stock2)]
        self.portfolio = Portfolio()

        self.portfolio = portfolio
        self.agent = Agent(action_dim=len(action_dict), state_dim=state_dim, abalance=10000, is_eval=False)


        self.balance = portfolio.balance
        self.t = t #user defined amount for dollar cost averaging

        self.switch = {
            'Hold': self.hold,

            # Buy % of stock value already in the portfolio
            'Buy_1share': self.buy_1share,
            'Buy_24p': self.buy_24p,
            'Buy_50p': self.buy_50p,
            'Buy_75p': self.buy_75p,
            'Buy_100p': self.buy_100p,
            # buy user defined cash amount
            'Buy_t': self.buy_t,

            # Sell % of stock value already in the portfolio
            'Sell_1share': self.sell_1share,
            'Sell_24p': self.sell_24p,
            'Sell_50p': self.sell_50p,
            'Sell_75p': self.sell_75p,
            'Sell_100p': self.sell_100p,
            # buy user defined cash amount
            'Sell_t': self.sell_t
        }

    def _initiate_state(self):
        # initiate an empty state
        return np.zeros((self.stock_dim, self.stock_dim))

    def take_actions_all_stocks(self):

        # 1 action per day per stock seems more plausible
        for index in range(len(self.stock_dim)):
            # NN_output is of type: [(action_index, stock_index), ()]
            action_index = self.NN_output[index][0]
            stock_index = self.NN_output[index][1]
            action_name = action_dict[action_index]

            # action_dict = {0: 'Hold', 2: 'Buy_1share', 3: 'Buy_24p', 4: 'Buy_50p', 5: 'Buy_100p',
            #                6: 'Sell_1share', 7: 'Sell_25p', 8: 'Sell_50p', 9: 'Sell_75p', 10: 'Sell100p',
            #                11: 'Buy_t', 12: 'Sell_t'  # user defined
            #                }


            # do one of the buy/sell actions per stock index
            self.switch[action_name](stock_index)


    def hold(self):
        logging.info('Hold')

    def buy_1share(self, stock_index):
        # somehow make the agent interract?
        # update portfolio


    # def buy(stock_index, amount):
    #     agent.balance -= stock_prices[t]
    #     agent.inventory.append(stock_prices[t])
    #     agent.buy_dates.append(t)
    #     logging.info('Buy:  ${:.2f}'.format(stock_prices[t]))



    def sell_t(self, stock_index):
        t = self.t



# [date_1, row1...]
# [date_2, row2..]
#num stocks