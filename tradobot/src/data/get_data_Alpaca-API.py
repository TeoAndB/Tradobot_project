# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

from processor_alpaca import *
from src.config import *
import click

@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    """ Runs Alpaca data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('getting datasets using Alpaca API')
    # # paper=True enables paper trading (simulated trading)
    # trading_client = TradingClient('PK94LDSMKV0KE60DNTZU', 'gCimhTZoS1XSpGRwjzQNlKE0S82vP3d6y6I7TO3D', paper=True)
    #
    # account = trading_client.get_account()

    # # stocks
    # TICKER_LIST_1 = ['HA', 'WBA', 'INCY']
    # TICKER_LIST_2 = ['HA', 'WBA', 'INCY', 'BIDU']
    # TICKER_LIST_3 = ['HA', 'WBA', 'INCY', 'BIDU', 'TCOM']
    # TICKER_LIST_4 = ['HA', 'WBA', 'INCY', 'AAPL']
    # TICKER_LIST_5 = ['HA', 'WBA', 'INCY', 'AAPL', 'COST']
    # TICKER_LIST_6 = ['HA', 'WBA', 'INCY', 'BIDU', 'TCOM', 'AAPL', 'COST']
    # TICKER_LIST_7 = ['BIDU', 'TCOM', 'AAPL', 'COST']
    #
    # # TICKER_LIST_lists = [TICKER_LIST_1, TICKER_LIST_2, TICKER_LIST_3, TICKER_LIST_4,
    # #                      TICKER_LIST_5, TICKER_LIST_6, TICKER_LIST_7]
    #
    # # Used in article: June 2014 – December 2019
    # TRAIN_START_DATE = '2016-02-01'
    # TRAIN_END_DATE = '2019-12-31'
    # # January 2020 – December 2021
    # TEST_START_DATE = '2020-01-01'
    # TEST_END_DATE = '2022-12-31'
    # TIME_INTERVAL = '15Min'

    i = 1
    for TICKER_LIST in TICKER_LIST_lists:
        logger.info(f'Retrieving training dataset {i}: {TICKER_LIST}...')
        alp_processor = AlpacaProcessor(API_KEY=ALPACA_API_KEY, API_SECRET=ALPACA_API_SECRET, API_BASE_URL=ALPACA_API_BASE_URL)
        df = alp_processor.download_data(TICKER_LIST, TRAIN_START_DATE, TRAIN_END_DATE, TIME_INTERVAL)
        df2 = alp_processor.clean_data(df)

        df2 = df2.rename(columns={"symbol": "tic"})

        df3 = alp_processor.add_technical_indicator(df2) #note there are some NaN values
        df4 = alp_processor.add_vix(df3)
        df5 = alp_processor.add_turbulence(df4).fillna(0)

        # save dataset
        df5.to_csv(f'{output_filepath}/train_stock_dataset_{i}.csv')
        print("Retrieved and cleaned dataset with technical indicators: ")
        print(df5.head(15))
        logger.info(f'Training Dataset {i} succesfully retireved. Cleaned, added stock indicators, vix and turbulence')

        i += 1

    for TICKER_LIST in TICKER_LIST_lists:
        logger.info(f'Retrieving test dataset {i}...')
        alp_processor = AlpacaProcessor(API_KEY=ALPACA_API_KEY, API_SECRET=ALPACA_API_SECRET, API_BASE_URL=ALPACA_API_BASE_URL)
        df = alp_processor.download_data(TICKER_LIST, TEST_START_DATE, TEST_END_DATE, TIME_INTERVAL)
        df2 = alp_processor.clean_data(df)
        df3 = alp_processor.add_technical_indicator(df2) #note there are some NaN values
        df4 = alp_processor.add_vix(df3)
        df5 = alp_processor.add_turbulence(df4).fillna(0)

        # save dataset
        df5.to_csv(f'{output_filepath}/test_stock_dataset_{i}.csv')
        print("Retrieved and cleaned dataset with technical indicators: ")
        print(df5.head(15))
        logger.info(f'Test Dataset {i} succesfully retireved. Cleaned, added stock indicators, vix and turbulence')

        i += 1

if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
