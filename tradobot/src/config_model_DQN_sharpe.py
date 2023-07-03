from __future__ import  annotations
from src.config_data import *


# CHANGE HERE
# predefined parameter that sets as the maximum amount of shares to trade.
hmax = 500
# liquid cash deposit
INITIAL_AMOUNT = 10000




buy_cost_pct = 0.001
sell_cost_pct = 0.001



# DATA PARAMETERS ####################################################
DATASET = "dataset1_1Day_HA-WBA-INCY.csv"
DATASET_INDEX = 1
NUM_STOCKS = 3

# DQN Params #####################
#NUM_EPOCHS = 10
NUM_EPOCHS = 10


GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9999
LEARNING_RATE = 0.0001
# BATCH_SIZE = 32
BATCH_SIZE = 68
TAU = 1e-3


# DATES #######################
TRAIN_START_DATE = '2014-07-01'
TRAIN_END_DATE = '2019-12-31'
# Used in article: January 2020 – December 2021
TEST_START_DATE = '2020-01-01'
TEST_END_DATE = '2021-12-31'

NUM_ACTIONS = 12

ACTIONS_DICTIONARY = {
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
    12: 'buy_1_share'
}
