from __future__ import  annotations
from src.config_data import *


# CHANGE HERE
# predefined parameter that sets as the maximum amount of shares to trade.
hmax = 100
# liquid cash deposit
initial_amount = 1000000



# TODO: add actual comission for these stock tickers - make a function
buy_cost_pct = 0.001
sell_cost_pct = 0.001



# DATA PARAMETERS ####################################################
TRAIN_DATASET = "dataset1_1Day_HA-WBA-INCY.csv"
DATASET_INDEX = 1


# DQN Params #####################
NUM_EPOCHS = 10


GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9999
LEARNING_RATE = 0.001
BATCH_SIZE = 32


# DATES #######################
TRAIN_START_DATE = '2014-07-01'
TRAIN_END_DATE = '2019-12-31'
# Used in article: January 2020 – December 2021
TEST_START_DATE = '2020-01-01'
TEST_END_DATE = '2021-12-31'

NUM_ACTIONS = 12

ACTION_DICTIONARY = {
    0: 'buy_0_1',
    1: 'buy_0_25',
    2: 'buy_0_50',
    3: 'buy_0_75',
    4: 'buy_1',
    5: 'sell_0_1',
    6: 'sell_0_25',
    7: 'sell_0_50',
    8: 'sell_0_75',
    9: 'sell_1',
    10: 'hold',
    11: 'sell_everything',
    12: 'buy_one_share'
}
