from __future__ import  annotations
from src.config_data import *

# predefined parameter that sets as the maximum amount of shares to trade.
hmax = 100
# liquid cash deposit
initial_amount = 1000000

# comission for trading
# TODO: add actual comission for these stock tickers - make a function
buy_cost_pct = 0.001
sell_cost_pct = 0.001

#scaling factor for reward, good for training
reward_scaling = 1e-4

# ?
print_verbosity = 5

# rebalance_window is the number of days to retrain the model
# validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)

# frequency training
rebalance_window = 120
validation_window = 120

# 1min frequency training
# rebalance_window is the number of time entries to retrain the model
# rebalance_window = 700 # entries from beginning to the end of the day
# validation_window = 700

# DATA PARAMETERS ####################################################
TRAIN_DATASET = "dataset1_1Day_HA-WBA-INCY.csv"
DATASET_INDEX = 1




# FINRL ENSEMBLE MODEL #############################################
'''
Only DDPG, A2C and PPO are examined during training. 
The Ensemble model class is instantated with more model parameters. ######

'''
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}



# DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
# #DDPG_PARAMS = {"batch_size": 64, "buffer_size": 10_000, "learning_rate":0.0005}
#
#
# # Model Parameters # ENSEMBLE #################
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
#DDPG_PARAMS = {"batch_size": 64, "buffer_size": 10_000, "learning_rate":0.0005}

TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}

A2C_model_kwargs = {
    'n_steps': 5,
    'ent_coef': 0.005,
    'learning_rate': 0.0007
}

PPO_model_kwargs = {
    "ent_coef": 0.01,
    "n_steps": 2048,
    "learning_rate": 0.00025,
    "batch_size": 128
}

DDPG_model_kwargs = {
    # "action_noise":"ornstein_uhlenbeck",
    "buffer_size": 10_000,
    "learning_rate": 0.0005,
    "batch_size": 64
}

TRAINED_MODEL_DIR = './models'
TENSORBOARD_LOG_DIR = './logs'

# DQN Params #####################
num_episode = 10

TRAIN_START_DATE = '2014-07-01'
TRAIN_END_DATE = '2019-12-31'
# Used in article: January 2020 – December 2021
TEST_START_DATE = '2020-01-01'
TEST_END_DATE = '2021-12-31'