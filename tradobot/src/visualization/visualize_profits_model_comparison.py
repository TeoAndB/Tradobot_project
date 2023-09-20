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
    reward_type_baseline = "baseline_results"

    data_type = "daily_frequency_data"
    # data_type = "minute_frequency_data"

    data_set = "training_last_epoch"
    # data_set = "validation_last_epoch"
    # data_set = "testing"

    models_compared = 'simpleDQN_and_simpleDQN_noLastLayerActivation'
    plot_title_models = 'simple DQN and simple DQN without last-layer activation'

    # Baseline ############################################################################################################
    name_dataset = 'HA-WBA-INCY'
    run_id = '2023-08-21_22_22'
    # no specifications since baseline

    profits_baseline_training_name = f'profit_per_year_last_epoch_training_dataset1_1Day_w14Lags_{name_dataset}_{run_id}.csv'
    profits_baseline_validation_name = f'profit_per_year_last_epoch_validation_dataset1_1Day_w14Lags_{name_dataset}_{run_id}.csv'
    df1 = pd.read_csv(f'{input_filepath}/{reward_type_baseline}/{data_type}/{profits_baseline_training_name}')
    df2 = pd.read_csv(f'{input_filepath}/{reward_type_baseline}/{data_type}/{profits_baseline_validation_name}')
    # Add a 'source' column to both dataframes
    df1['source'] = 'training'
    df2['source'] = 'validation'
    df1 = df1.rename(columns={'year_training': 'year', 'yearly_profit_training': 'yearly_profit'})
    df2 = df2.rename(columns={'year_validation': 'year', 'yearly_profit_validation': 'yearly_profit'})
    df_combined = pd.concat([df1, df2])
    df_combined = df_combined.sort_values(by='year')

    year_list_baseline = df_combined['year'].tolist()
    yearly_profit_list_baseline = df_combined['yearly_profit'].tolist()
    source_list_baseline = df_combined['source'].tolist()

    del df1
    del df2

    # DQN Model 1 ############################################################################################

    name_dataset = 'HA-WBA-INCY'
    run_ids = ['2023-09-03_20_35', '2023-09-03_19_05', '2023-09-03_18_23']
    specifications = 'simple_DQN_1Layer_30Epochs'

    profits_list_DQN_runs = []
    source_list_DQN_runs = []
    years_list_DQN_runs = []

    for run_id in run_ids:
        profits_DQN_training_name = f'profit_per_year_last_epoch_training_dataset1_1Day_w14Lags_{name_dataset}_{run_id}_{specifications}.csv'
        profits_DQN_validation_name = f'profit_per_year_last_epoch_validation_dataset1_1Day_w14Lags_{name_dataset}_{run_id}_{specifications}.csv'
        df1 = pd.read_csv(f'{input_filepath}/{reward_type}/{data_type}/{profits_DQN_training_name}')
        df2 = pd.read_csv(f'{input_filepath}/{reward_type}/{data_type}/{profits_DQN_validation_name}')

        df1['source'] = 'training'
        df2['source'] = 'validation'
        df1 = df1.rename(columns={'year_training': 'year', 'balance_per_year_training': 'yearly_balance', 'profit_per_year_training': 'yearly_profit'})
        df2 = df2.rename(columns={'year_validation': 'year', 'balance_per_year_validation': 'yearly_balance','profit_per_year_validation': 'yearly_profit'})
        df_combined = pd.concat([df1, df2])
        df_combined = df_combined.sort_values(by='year')
        del df1
        del df2

        yearly_profit_list_DQN = df_combined['yearly_profit'].tolist()
        profits_list_DQN_runs.append(yearly_profit_list_DQN)
        # for verifications
        source_list_DQN = df_combined['source'].tolist()
        source_list_DQN_runs.append(source_list_DQN)

        year_list_DQN = df_combined['year'].tolist()
        years_list_DQN_runs.append(year_list_DQN)

    # DQN Model 2 ############################################################################################

    name_dataset = 'HA-WBA-INCY'
    run_ids = ['2023-09-13_21_49', '2023-09-14_15_59', '2023-09-18_21_19']
    specifications = 'DQN_simple_nosoftmax_tanh_30Epochs'

    profits_list_DQN2_runs = []
    source_list_DQN2_runs = []
    years_list_DQN2_runs = []

    for run_id in run_ids:
        profits_DQN2_training_name = f'profit_per_year_last_epoch_training_dataset1_1Day_w14Lags_{name_dataset}_{run_id}_{specifications}.csv'
        profits_DQN2_validation_name = f'profit_per_year_last_epoch_validation_dataset1_1Day_w14Lags_{name_dataset}_{run_id}_{specifications}.csv'
        df1 = pd.read_csv(f'{input_filepath}/{reward_type}/{data_type}/{profits_DQN2_training_name}')
        df2 = pd.read_csv(f'{input_filepath}/{reward_type}/{data_type}/{profits_DQN2_validation_name}')

        df1['source'] = 'training'
        df2['source'] = 'validation'
        df1 = df1.rename(columns={'year_training': 'year', 'balance_per_year_training': 'yearly_balance', 'profit_per_year_training': 'yearly_profit'})
        df2 = df2.rename(columns={'year_validation': 'year', 'balance_per_year_validation': 'yearly_balance','profit_per_year_validation': 'yearly_profit'})
        df_combined = pd.concat([df1, df2])
        df_combined = df_combined.sort_values(by='year')
        del df1
        del df2

        yearly_profit_list_DQN2 = df_combined['yearly_profit'].tolist()
        profits_list_DQN2_runs.append(yearly_profit_list_DQN2)
        # for verifications
        source_list_DQN2 = df_combined['source'].tolist()
        source_list_DQN2_runs.append(source_list_DQN2)

        year_list_DQN2 = df_combined['year'].tolist()
        years_list_DQN2_runs.append(year_list_DQN2)

    #############################################################################################################################################

    print(f'year_list_baseline is \n{year_list_baseline}\n')
    print(f'yearly_profit_list_baseline is \n{yearly_profit_list_baseline}\n')
    print(f'source_list_baseline is \n{source_list_baseline}\n')

    print(f'years_list_DQN_runs is \n{years_list_DQN_runs}\n')
    print(f'profits_list_DQN_runs is \n{profits_list_DQN_runs}\n')
    print(f'source_list_DQN_runs is \n{source_list_DQN_runs}\n')

    print(f'years_list_DQN2_runs is \n{years_list_DQN_runs}\n')
    print(f'profits_list_DQN2_runs is \n{profits_list_DQN_runs}\n')
    print(f'source_list_DQN_runs is \n{source_list_DQN_runs}\n')

    if year_list_baseline == years_list_DQN_runs[0] and year_list_baseline == years_list_DQN2_runs[0]:
        print("The years are in the correct order for all models.")
    else:
        print("\033[91mThe years are NOT in the correct order for all models.\033[0m")

    if source_list_baseline == source_list_DQN_runs[0] and source_list_baseline == source_list_DQN2_runs[0]:
        print("The training and validation sources are in the correct order for all models.")
    else:
        print("\033[91mThe training and validation sources are NOT in the correct order for all models.\033[0m")

    # Plotting ##################################################################################################
    # Average and standard deviation for DQN
    avg_profits_DQN = np.mean(profits_list_DQN_runs, axis=0)
    std_profits_DQN = np.std(profits_list_DQN_runs, axis=0)

    # Average and standard deviation for DQN2
    avg_profits_DQN2 = np.mean(profits_list_DQN2_runs, axis=0)
    std_profits_DQN2 = np.std(profits_list_DQN2_runs, axis=0)

    # Define a function to get colors based on source
    def get_colors_for_source(source_list):
        return ['cyan' if s == 'training' else 'orange' for s in source_list]

    # Plotting
    plt.figure(figsize=(12, 8))

    colors_baseline = get_colors_for_source(source_list_baseline)

    # Baseline
    line_baseline, = plt.plot(year_list_baseline, yearly_profit_list_baseline, '--r', label='Baseline: Balanced Agent')
    dots_baseline = []
    for year, profit, color in zip(year_list_baseline, yearly_profit_list_baseline, colors_baseline):
        dot, = plt.plot(year, profit, 'o', markersize=8, markerfacecolor=color, markeredgewidth=0.5,
                        markeredgecolor='black')
        dots_baseline.append(dot)

    # Simple DQN with standard deviation
    line_DQN, = plt.plot(year_list_baseline, avg_profits_DQN, '-b', label='Simple DQN: Avg of 3 Training and Validation Runs')
    dots_DQN = []
    for year, profit, color in zip(year_list_baseline, avg_profits_DQN, colors_baseline):
        dot, = plt.plot(year, profit, 'o', markersize=8, markerfacecolor=color, markeredgewidth=0.5,
                        markeredgecolor='blue')
        dots_DQN.append(dot)
    plt.fill_between(year_list_baseline, avg_profits_DQN - std_profits_DQN, avg_profits_DQN + std_profits_DQN,
                     color='blue', alpha=0.2)

    # DQN2 with standard deviation
    line_DQN2, = plt.plot(year_list_baseline, avg_profits_DQN2, '-g', label='Simple DQN no Activation: Avg of 3 Training and Validation Runs')
    dots_DQN2 = []
    for year, profit, color in zip(year_list_baseline, avg_profits_DQN2, colors_baseline):
        dot, = plt.plot(year, profit, 'o', markersize=8, markerfacecolor=color, markeredgewidth=0.5,
                        markeredgecolor='green')
        dots_DQN2.append(dot)
    plt.fill_between(year_list_baseline, avg_profits_DQN2 - std_profits_DQN2, avg_profits_DQN2 + std_profits_DQN2,
                     color='green', alpha=0.2)

    # Custom legend handles for training and validation dots
    legend_train = plt.Line2D([0], [0], marker='o', color='w', label='Training', markersize=8, markerfacecolor='cyan',
                              linestyle='None')
    legend_validation = plt.Line2D([0], [0], marker='o', color='w', label='Validation', markersize=8,
                                   markerfacecolor='orange', linestyle='None')

    # Additional plotting parameters
    plt.title(f'Yearly Profits - Model Comparison for the {name_dataset} Portfolio Dataset: \n{plot_title_models}')
    plt.xlabel('Year')
    plt.ylabel('Profits')
    plt.legend(handles=[line_baseline, line_DQN, line_DQN2, legend_train, legend_validation], loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./reports/model_performance_figures/comparison_{models_compared}_on_{name_dataset}_dataset_train2014-2019__val2020-2022.png')
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
