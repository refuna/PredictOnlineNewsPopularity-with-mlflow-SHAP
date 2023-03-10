# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv

import pandas as pd
import numpy as np
import sklearn.model_selection as ms

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # read data
    df = pd.read_csv(input_filepath, sep=', ', engine='python')

    # Define the target variable
    df['popularity'] = df['shares'].apply(lambda x: 1 if x > 1400 else 0)

    # drop columns that are not needed
    df = df.drop(['url', 'timedelta'], axis=1)

    # split into train and test
    train, test = ms.train_test_split(df, test_size=0.2, random_state=42)

    # save data
    train.to_csv(output_filepath + '/train.csv', index=False)
    test.to_csv(output_filepath + '/test.csv', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
