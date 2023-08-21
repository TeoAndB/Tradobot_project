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



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        This script mostly focuses on splitting the data into training and validation sets.
    """

    reward_type = "reward_portfolio_return"
    # reward_type = "reward_sharpe_ratio"

    data_type = "daily_frequency_data"
    # data_type = "minute_frequency_data"

    data_set = "training_last_epoch"
    # data_set = "validation_last_epoch"
    # data_set = "testing"

    losses_data_1 = "loss_per_epoch_2023-08-21_22_45_.csv"
    df1 = pd.read_csv(f'{input_filepath}/{reward_type}/{data_type}/{losses_data_1}')

    losses_data_2 = "loss_per_epoch_2023-08-21_22_56_w_memory_reset.csv"
    df2 = pd.read_csv(f'{input_filepath}/{reward_type}/{data_type}/{losses_data_2}')

    losses_data_3 = "loss_per_epoch_2023-08-21_23_12_w_memory_reset.csv"
    df3 = pd.read_csv(f'{input_filepath}/{reward_type}/{data_type}/{losses_data_3}')

    dataset_name = 'HA-WBA-INCY'

    # Opacity values for differentiation
    alphas = [1.0, 0.7, 0.4]

    # Plotting
    for idx, df in enumerate([df1, df2, df3]):
        plt.plot(df['epoch_no_training'], df['loss_training'], label=f'Training {idx + 1}', color='blue',
                 alpha=alphas[idx])
        plt.plot(df['epoch_no_validation'], df['loss_validation'], '--', label=f'Validation {idx + 1}', color='orange',
                 alpha=alphas[idx])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses for Different Model Training Experiments')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'./reports/explainability_figures/{reward_type}/{data_type}/{data_set}/different_LossCurves_plots_for_{dataset_name}.png')
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
