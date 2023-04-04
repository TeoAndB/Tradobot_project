import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.config_data import *
from src.config_model import *
import pandas as pd
import glob
import os
from src.models.DRL_model_stablebaselines import *


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        This script mostly focuses on splitting the data into training and validation sets.
    """
    logger = logging.getLogger(__name__)

    logger.info('Retrieving Data')

    INDICATORS = ["macd",
            "boll_ub",
            "boll_lb",
            "rsi_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",]

    # TICKER_LIST_1 dataset
    processed = pd.read_csv(f'{input_filepath}/{TRAIN_DATASET}')
    processed.rename(columns={'timestamp':'date'}, inplace=True)
    print(processed)
    stock_dimension = len(processed.tic.unique())
    INDICATORS = processed.tic.unique().tolist()
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


    logger.info('Training Model')

    env_kwargs = {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "buy_cost_pct": buy_cost_pct,
        "sell_cost_pct": sell_cost_pct,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": reward_scaling,
        "print_verbosity": print_verbosity
    }

    # rebalance_window = 63  # rebalance_window is the number of days to retrain the model
    # validation_window = 63  # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)

    DDPG_model_kwargs = {
        # "action_noise":"ornstein_uhlenbeck",
        "buffer_size": 10_000,
        "learning_rate": 0.0005,
        "batch_size": 64
    }

    timesteps_dict = {
                      'ddpg': 10_000
                      }
    #
    agent = DRLAgent(df=processed,
                      train_period=(TRAIN_START_DATE, TRAIN_END_DATE),
                      val_test_period=(TEST_START_DATE, TEST_END_DATE),
                      rebalance_window=rebalance_window,
                      validation_window=validation_window,
                      **env_kwargs)

    df_summary = agent.run_strategy(DDPG_model_kwargs,
                                      timesteps_dict)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()