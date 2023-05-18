import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models.FinRL_EnsembleModel import *


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Trains the FinRL model.
    """
    logger = logging.getLogger(__name__)

    logger.info('Retrieving Data')

    # INDICATORS = ["macd",
    #         "boll_ub",
    #         "boll_lb",
    #         "rsi_30",
    #         "dx_30",
    #         "close_30_sma",
    #         "close_60_sma",]

    # TICKER_LIST_1 dataset
    processed = pd.read_csv(f'{input_filepath}/{TRAIN_DATASET}')
    processed.rename(columns={'timestamp':'date'}, inplace=True)
    print(processed)
    stock_dimension = len(processed.tic.unique())
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



    timesteps_dict = {'a2c': 10_000,
                      'ppo': 10_000,
                      'ddpg': 10_000
                      }

    #
    agent = DRLEnsembleAgent(df=processed,
                      train_period=(TRAIN_START_DATE, TRAIN_END_DATE),
                      val_test_period=(TEST_START_DATE, TEST_END_DATE),
                      rebalance_window=rebalance_window,
                      validation_window=validation_window,
                      **env_kwargs)

    df_summary = agent.run_ensemble_strategy(A2C_model_kwargs,
                                            PPO_model_kwargs,
                                            DDPG_model_kwargs,
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