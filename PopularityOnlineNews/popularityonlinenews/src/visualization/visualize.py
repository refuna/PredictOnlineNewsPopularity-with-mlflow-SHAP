# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import joblib
import os

import eli5
from eli5.sklearn import PermutationImportance

import shap

@click.command()
@click.argument('test_filepath', type=click.Path(exists=True))
@click.argument('models_dir', type=click.Path(exists=True))

def main(test_filepath, models_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #  read the data
    data = pd.read_csv(test_filepath)
    if 'popularity' in data.columns:
        X = data.drop('popularity', axis=1)
        y = data['popularity']
    else:
        X = data

    # load the model
    if os.path.exists(models_dir + '/rf.joblib'):
        rf = joblib.load(models_dir + '/rf.joblib')
        logger.info('Model successfully loaded')
    else:
        raise Exception('No model found, please train the model first')

    # predict
    perm = PermutationImportance(rf, random_state=47).fit(X, y)
    eli5.show_weights(perm, feature_names = X.columns.tolist())

    print("done")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()