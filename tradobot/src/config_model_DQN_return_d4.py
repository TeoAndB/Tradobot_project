from __future__ import  annotations
from src.config_data import *


# CHANGE HERE
# predefined parameter that sets as the maximum amount of shares to trade.
hmax = 1000
# liquid cash deposit
INITIAL_AMOUNT = 10000
# please work

# DATA PARAMETERS ####################################################
DATASET = "dataset4_1Day_w14Lags_HA-WBA-INCY-AAPL.csv"
DATASET_INDEX = 4
NUM_STOCKS = 4
TIME_LAG = 14

# DQN Params #####################
#NUM_EPOCHS = 10
NUM_EPOCHS = 35
# Define the weight decay value
# WEIGHT_DECAY = 1e-4
NUM_SAMPLING = 2

GAMMA = 0.80 # change to 1
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.90
#EPSILON_DECAY = 0.9900
# LEARNING_RATE = 0.0001
# LEARNING_RATE = 1e-8
LEARNING_RATE = 1e-4
# BATCH_SIZE = 16
#BATCH_SIZE = 32
BATCH_SIZE = 128
TAU = 1e-3 # higher tau, higher convergence but more training instability
ALPHA_CQL = 1.0 #trade-off factor for the CQL regularizer


# DATES #######################
# TRAIN_START_DATE = '2014-07-01'
# TRAIN_END_DATE = '2019-12-31'
# # Used in article: January 2020 – December 2021
# TEST_START_DATE = '2020-01-01'
# TEST_END_DATE = '2021-12-31'

NUM_ACTIONS = 13

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


