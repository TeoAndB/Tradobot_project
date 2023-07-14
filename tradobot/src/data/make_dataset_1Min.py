# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import glob
import os
import shutil
from src.features.preprocessor_FinRL_1Min import *
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

    # TODO: Add this in data config list
    TICKER_LIST = ['AMZN','MSFT','SPY']

    i = 1
    df_list = []

    # List to store unique date entries
    unique_dates = []

    df = pd.DataFrame()

    for TICKER in TICKER_LIST:
        df = pd.read_csv(f'{input_filepath}/{TICKER}_FirstRateDatacom1.txt', sep=",", header=None)
        # add columns
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        # Extract the date column
        dates = pd.to_datetime(df['date']).tolist()

        # Append unique date entries to the list
        unique_dates.extend([date for date in dates if date not in unique_dates])

    unique_dates.sort()
    print(f'unique dates are {unique_dates[:30]}')

    for TICKER in TICKER_LIST:
        df = pd.read_csv(f'{input_filepath}/{TICKER}_FirstRateDatacom1.txt', sep=",", header=None)
        # add columns
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        ## Add technical indicators ##############################


        df['tic'] = TICKER

        fe = FeatureEngineer(use_technical_indicator=True,
                             tech_indicator_list=INDICATORS,
                             use_vix=False,
                             use_turbulence=False,
                             user_defined_feature=False)

        processed = fe.preprocess_data(df)

        processed = processed.copy()
        processed = processed.fillna(0)
        processed = processed.drop('turbulence', axis=1)
        # processed = processed.replace(np.inf, 0)
        print("Successfully added technical indicators, but no vix or turbulence")



        ## Fill in missing dates from unique_dates ################

        # Set 'date' column as datetime index
        processed['date'] = pd.to_datetime(processed['date'])
        processed.set_index('date', inplace=True)

        # Create a new DataFrame with the missing dates
        missing_dates_df = pd.DataFrame(index=pd.to_datetime(unique_dates), columns=processed.columns)

        # Concatenate processed DataFrame and missing_dates_df
        processed = processed.combine_first(missing_dates_df)

        # Sort the DataFrame by date
        processed.sort_index(inplace=True)

        # Forward fill missing values within each unique date
        processed.groupby(processed.index.date).fillna(method='ffill', inplace=True)

        # Fill missing timestamp values with values from previous dates

        # Reset the index and move 'date' column to the first position
        processed.reset_index(inplace=True)

        ###############################################################

        # replace 0.0 with NaN values
        processed.replace(0, np.nan, inplace=True)

        # Propagate last valid observation forward to next valid.
        processed.fillna(method="ffill", inplace=True)

        # Have to rename one column:
        processed.rename(columns={'index': 'date'}, inplace=True)

        processed = processed.set_index(['date','tic'])

        # Remove rows with duplicate index values
        processed = processed[~processed.index.duplicated(keep='first')]

        print(f'The dataset for {TICKER} dataset \n {processed.head(15)}')
        processed.to_csv(f'data/processed/dataset_1Min_{TICKER}.csv')

        # append each dataset to a list
        df_list.append(processed)

    # concatenate datasets with filled timestamp values
    cols_idx = ['date', 'tic']
    keys = list(set(df_list[0].columns) - set(cols_idx))

    df_merged = pd.concat(df_list).sort_values(by=["date","tic"])

    # Trim the dataset at the date where there is consistent data:
    df_merged = df_merged.loc[('2019-01-02 09:30:00',):]

    print(f'Merged Dataset created for {TICKER_LIST} is \n{df_merged.head(10)}')

    name_dataset = '-'.join(TICKER_LIST)
    df_merged.to_csv(f'{output_filepath}/dataset_1Min_{name_dataset}.csv')
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
