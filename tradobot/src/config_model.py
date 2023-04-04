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
rebalance_window = 63
# validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)
validation_window = 63

DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}

TRAIN_DATASET = "train_stock_dataset_1.csv"