# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import glob
import os
import shutil


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

    # reading all csv files in data/raw
    csv_files = glob.glob(os.path.join(input_filepath, "*.csv"))

    for f in csv_files:
        # read the csv file
        df = pd.read_csv(f)

        # print the location and filename
        print('Downloaded and Cleaned Data Location:', f)
        print('Downloaded and Cleaned Data File Name:', f.split("\\")[-1])


        print('Processed Data Location:', output_filepath)
        print('Processed Data File Name:', f'{output_filepath}/data_Alpaca.csv')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
