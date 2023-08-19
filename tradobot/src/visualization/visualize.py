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

    # data_set = "training_last_epoch"
    # data_set = "validation_last_epoch"
    data_set = "testing"

    explainability_data = "testing_explainability_1_dataset1_1Day_w14Lags_HA-WBA-INCY_2023-08-17_22_12_HuberLoss_L1L2reg_wLSTM_article_dates.csv"
    df = pd.read_csv(f'{input_filepath}/{reward_type}/{data_type}/{data_set}/{explainability_data}')

    dataset_name = "HA-WBA-INCY"
    tic_colors = {
        'HA': 'magenta',
        'INCY': 'cyan',
        'WBA': 'yellow'
    }

    logger = logging.getLogger(__name__)
    logger.info(f'Creating explainability plot for file:\n {explainability_data}')
    df['Dates'] = pd.to_datetime(df['Dates'])

    print(df.head(10))

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(14, 7))

    # List of tics
    tics = df['TIC'].unique()
    # tics = ["HA"]

    # Define intensity for green (buy) based on ranking
    green_intensity = {
        'buy_1_share': mcolors.to_rgba("green", 0.1),
        'buy_0_1': mcolors.to_rgba("green", 0.2),
        'buy_0_25': mcolors.to_rgba("green", 0.4),
        'buy_0_50': mcolors.to_rgba("green", 0.6),
        'buy_0_75': mcolors.to_rgba("green", 0.8),
        'buy_1': mcolors.to_rgba("green", 1)
    }

    # Define intensity for red (sell) based on ranking
    red_intensity = {
        'sell_0_1': mcolors.to_rgba("red", 0.1),
        'sell_0_25': mcolors.to_rgba("red", 0.2),
        'sell_0_50': mcolors.to_rgba("red", 0.4),
        'sell_0_75': mcolors.to_rgba("red", 0.6),
        'sell_1': mcolors.to_rgba("red", 0.8),
        'sell_everything': mcolors.to_rgba("red", 1)
    }

    # Plot each tic's data
    for tic in tics:
        sub_df = df[df['TIC'] == tic]
        ax.plot(sub_df['Dates'], sub_df['Closing_Price'], label=tic, color=tic_colors[tic], zorder=1)

        # Add green triangles for buy, red triangles for sell, blue dots for hold
        for _, row in sub_df.iterrows():
            if "buy" in str(row['Actions']):
                ax.scatter(row['Dates'], row['Closing_Price'], color=green_intensity[row['Actions']], marker='^', s=100,
                           zorder=2)
            elif "sell" in str(row['Actions']):
                ax.scatter(row['Dates'], row['Closing_Price'], color=red_intensity[row['Actions']], marker='v', s=100,
                           zorder=2)
            elif row['Actions'] == "hold":
                ax.scatter(row['Dates'], row['Closing_Price'], color="blue", marker='o', s=50, zorder=2)

    # Set x-ticks to display only a limited number of dates
    unique_dates = df['Dates'].unique()
    num_dates = len(unique_dates)
    selected_ticks = unique_dates[np.linspace(0, num_dates - 1, 5, dtype=int)]

    ax.set_xticks(selected_ticks)

    # Set labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Price per share in $")
    ax.set_title(f'Agent trading decisions for {data_set} period according to the \nclosing price evolution of {dataset_name} stocks')

    legend_elements = []
    for tic in tics:
        legend_elements.append(Line2D([0], [0], color=tic_colors[tic], lw=2, label=tic))
    legend_elements_symbols = [
                           Line2D([0], [0], marker='^', color='w', label='Buy', markersize=8, markerfacecolor='green'),
                           Line2D([0], [0], marker='v', color='w', label='Sell', markersize=8, markerfacecolor='red'),
                           Line2D([0], [0], marker='o', color='w', label='Hold', markersize=8, markerfacecolor='blue')]
    # Add custom legend entries for the buy, sell, and hold actions, and for TICs
    legend_elements.extend(legend_elements_symbols)

    ax.legend(handles=legend_elements, loc='upper left')

    # Add a grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.set_facecolor('#e0e0e0')  # This sets a light gray background color

    plt.tight_layout()
    plt.savefig(f'./reports/explainability_figures/{reward_type}/{data_type}/{data_set}/explainability_plots_for_{dataset_name}.png')

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
