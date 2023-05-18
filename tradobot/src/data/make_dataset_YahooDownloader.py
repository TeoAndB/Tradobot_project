# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import glob
import os
import shutil
from src.data.YahooDownloader import *
from src.features.preprocessor_FinRL import *
from src.config_data import *


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        This script mostly focuses on splitting the data into training and validation sets.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    i = 1
    for TICKER_LIST in TICKER_LIST_lists:
        df = YahooDownloader(start_date=TRAIN_START_DATE,
                             end_date=TEST_END_DATE,
                             ticker_list=TICKER_LIST).fetch_data()
        print("DATA RETRIEVED.")
        fe = FeatureEngineer(use_technical_indicator=True,
                             tech_indicator_list=INDICATORS,
                             use_turbulence=True,
                             user_defined_feature=False)

        processed = fe.preprocess_data(df)
        processed = processed.copy()
        processed = processed.fillna(0)
        processed = processed.replace(np.inf, 0)
        print("Successfully added technical indicators and turbulence")
        print(f'Data retrieved for {TICKER_LIST} is \n{processed.sample(5)}')
        print('Summary of data:')
        print(processed.describe())
        name_dataset = '-'.join(TICKER_LIST)
        processed.to_csv(f'data/processed/dataset{i}_1Day_{name_dataset}.csv')
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
