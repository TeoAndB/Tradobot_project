# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import glob
import os
import shutil
from src.config_data import *
from src.config_model_DQN_return import *



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        This script mostly focuses on splitting the data into training and validation sets.
    """

    # Read the data from CSV files
    df_agent_rl = pd.read_csv(f'{input_filepath}/reward_portfolio_return/daily_frequency_data/validation_last_epoch/validation_explainability_dataset1_1Day_w14Lags_HA-WBA-INCY_2023-09-27_21_36_DQN_simple_ReLu_30Epochs_noreset.csv')
    df_buy_and_hold = pd.read_csv(f'{input_filepath}/baseline_results/minute_frequency_data/balanced_agent_validation_period_dataset1_1Day_w14Lags_HA-WBA-INCY_2023-09-29_14_01_.csv')

    data_type = 'daily_frequency_data'
    dataset_name = 'HA-WA-INCY'
    # Convert the 'Dates' column to datetime format
    df_agent_rl['Dates'] = pd.to_datetime(df_agent_rl['Dates'])
    df_buy_and_hold['Dates'] = pd.to_datetime(df_buy_and_hold['Dates'])

    # Filter the dataframes for the desired date range
    df_agent_rl = df_agent_rl[(df_agent_rl['Dates'].dt.year >= 2020) & (df_agent_rl['Dates'].dt.year <= 2022)]
    df_buy_and_hold = df_buy_and_hold[
        (df_buy_and_hold['Dates'].dt.year >= 2020) & (df_buy_and_hold['Dates'].dt.year <= 2022)]

    # Drop duplicates Dates keeping the last occurrence (if your data has multiple rows for the same date)
    df_agent_rl = df_agent_rl.drop_duplicates(subset='Dates', keep='last')
    df_buy_and_hold = df_buy_and_hold.drop_duplicates(subset='Dates', keep='last')

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df_agent_rl['Dates'], df_agent_rl['total_balance']/INITIAL_AMOUNT, label='simple-DQN Agent')
    plt.plot(df_buy_and_hold['Dates'], df_buy_and_hold['total_balance']/INITIAL_AMOUNT, label='Buy and Hold')

    # Rotating x-axis ticks by 45 degrees
    plt.xticks(rotation=45)

    plt.xlabel('Dates')
    plt.ylabel('Profit')
    plt.title('Cumulated Profits Comparison between simple-DQN Agent and Buy and Hold on Validation Dataset (2020-2022)')
    plt.legend()
    plt.grid()

    plt.tight_layout()  # Ensure that everything fits without overlapping
    plt.savefig(
        f'./reports/figures/baseline_models/{data_type}/balanced_agent_and_RL_agent_profits_for_{dataset_name}.png')

    plt.show()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()