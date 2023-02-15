import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    df = pd.read_csv(input_filepath)

    # Keep only the relevant columns
    important_features = ['kw_avg_avg','weekday_is_saturday', 'self_reference_avg_sharess', 'average_token_length',
                      'n_unique_tokens', 'avg_positive_polarity', 'num_hrefs', 'global_subjectivity', 'num_videos']


    # Remove the outliers (only for training data)
    if 'train' in input_filepath:
        df = df[df['num_hrefs'] < 250]

    # Remove rows with missing values
    if 'train' in input_filepath:
        df = df.dropna()

    df.reset_index(drop=True, inplace=True)
    scaler_filename = os.path.split(output_filepath)[0] + '/scaler.joblib'

    # Scale the data
    if 'train' in input_filepath:
        scaler = StandardScaler()
        scaler.fit(df[important_features])
        joblib.dump(scaler, scaler_filename)
        logger.info('Scaling successfully done and saved')
    else:
        if os.path.exists(scaler_filename):
            scaler = joblib.load(scaler_filename)
            logger.info('Scaling successfully loaded')
        else:
            raise Exception('No scaler found, please train the model first')

    scaled_df = pd.DataFrame(scaler.transform(df[important_features]), columns=important_features)

    if 'popularity' in df.columns:
        scaled_df['popularity'] = df['popularity']

    # Save the data
    scaled_df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()