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

    for TICKER in TICKER_LIST:
        df = pd.read_csv(f'{input_filepath}/{TICKER}_FirstRateDatacom1.txt', sep=",", header=None)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df['tic'] = TICKER

        fe = FeatureEngineer(use_technical_indicator=True,
                             tech_indicator_list=INDICATORS,
                             use_vix=False,
                             use_turbulence=False,
                             user_defined_feature=False)

        processed = fe.preprocess_data(df)
        processed = processed.copy()
        # processed = processed.fillna(0)
        # processed = processed.replace(np.inf, 0)
        print("Successfully added technical indicators, but no vix or turbulence")

        df_list.append(processed)

    df = pd.concat(df_list, axis=0, ignore_index=False).copy()
    df.sort_values(by='date', inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)

    print(f'dataset created is \n{df.head(10)}')


    print(f'Data retrieved for {TICKER_LIST} is \n{processed.sample(5)}')
    print('Summary of data:')
    print(processed.describe())
    name_dataset = '-'.join(TICKER_LIST)
    processed.to_csv(f'data/processed/dataset{i}_1Min_{name_dataset}.csv')
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
