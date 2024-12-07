
from utils import load_config, load_data
from typing import List
import pandas as pd
from log_file import log_message, set_up_logs
import argparse
import os


# Set up logging for the script
logger = set_up_logs(__name__, 'preprocess.log')

# Define paths for raw and processed data folders
raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/raw')
processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/processed')

# Get the current working directory to use for default config file path
config_ = os.getcwd()


# Function to read data from a Parquet file
def read_data(file_path: str) -> pd.DataFrame: 
    try: 
        data = pd.read_parquet(file_path)  # Load Parquet file into a DataFrame
        log_message(logger, 'info', f'Successfully loaded Parquet file: {file_path}')  # Log success
        return data
    except FileNotFoundError:
        log_message(logger, 'error', f'File not found at {file_path}')  # Log error if file not found
    except Exception as e: 
        log_message(logger, 'error', f'Error loading data from {file_path}: {e}')  # Log any other errors


# Function to handle discrete variable transformations
def discrete_transformation(df: pd.DataFrame, discrete_columns: List[str]) -> pd.DataFrame:
    try: 
        discrete = df.copy()  # Create a copy of the original DataFrame
        for col in discrete_columns:  # Loop through each discrete column
            modal = discrete[col].mode()[0]  # Find the most frequent value (mode)
            discrete[col] = discrete[col].fillna(modal)  # Replace missing values with the mode
        discrete[discrete_columns[1]] = discrete[discrete_columns[1]].replace(0, 1)  # Replace 0 with 1 in the second column
        log_message(logger, 'info', f'Discrete variables transformed for columns: {discrete_columns}')  # Log success
        return discrete
    except KeyError: 
        log_message(logger, 'error', f'Column(s) not found: {discrete_columns}')  # Log error if column not found
    except Exception as e: 
        log_message(logger, 'error', f'Error during discrete transformation: {e}')  # Log any other errors


# Function to handle object (categorical) variable transformations
def object_transformation(df: pd.DataFrame, object_columns: List[str], target_col: str) -> pd.DataFrame:
    try:
        data = df.copy()  # Create a copy of the DataFrame

        # Loop through each object column
        for col in object_columns:
            var = data.groupby(col)[target_col].mean().index[1]  # Get the second most frequent value
            data[col] = data[col].fillna(var)  # Fill missing values with the second most frequent value

        # Create a label mapper for the categorical columns
        for col in object_columns:
            label_mapper = {var: idx for idx, (var, _) in enumerate(data.groupby(col)[target_col].mean().items())}
            data[col] = data[col].map(label_mapper)  # Map the categorical values to numeric labels

        log_message(logger, 'info', f'Object variables transformed for columns: {object_columns}')  # Log success
        return data
    except KeyError: 
        log_message(logger, 'error', f'Column(s) not found: {object_columns}')  # Log error if column not found
    except Exception as e: 
        log_message(logger, 'error', f'Error during object transformation: {e}')  # Log any other errors


# Function to perform feature engineering on the data
def feature_engineering(data: pd.DataFrame, feature_cols: List[str], date_cols: List[str], discrete_cols: List[str]) -> pd.DataFrame:
    try:
        df = data.copy()  # Create a copy of the DataFrame

        # Remove rows with invalid or zero values in specific columns
        df = df[df[feature_cols[0]] != 0]
        df = df[df[feature_cols[3]] != 0]
        df = df[df[feature_cols[1]] != df[feature_cols[2]]]

        # Create new features for duration, fare per distance, and fare per passenger
        df['duration'] = (df[date_cols[1]] - df[date_cols[0]]).dt.total_seconds() / 60
        df['fare_per_dist'] = df[feature_cols[3]] - df[feature_cols[1]]
        df['fare_per_passenger'] = df[feature_cols[3]] - df[discrete_cols[1]]

        df.drop(labels=date_cols, axis=1, inplace=True)  # Drop the original date columns
        log_message(logger, 'info', f'Feature engineering completed for columns: {feature_cols}, date columns: {date_cols}, and discrete columns: {discrete_cols}')  # Log success
        return df 
    except KeyError:
        log_message(logger, 'error', f'Column(s) not found: {feature_cols}, {date_cols}, {discrete_cols}')  # Log error if column not found
    except Exception as e:
        log_message(logger, 'error', f'Error during feature engineering: {e}')  # Log any other errors


# Main function to load, process, and save data
def main(config_file: str, raw_data: str, processed_data: str): 
    try:
        # Load configuration and get the column names for different data types
        config = load_config(config_file)['preprocess']
        numeric_columns = config['Numerical_Variable']
        object_columns = config['Categorical_Variable']
        discrete_columns = config['Discrete_Variable']
        date_columns = config['Datetime_Variable']
        target_column = config['Target_Variable']

        # Read the raw data
        taxi_df = read_data(raw_data)

        # Perform discrete transformation
        taxi_processed_dis = discrete_transformation(taxi_df, discrete_columns=discrete_columns)

        # Perform object transformation
        taxi_processed_obj = object_transformation(taxi_processed_dis, object_columns=object_columns, target_col=target_column)
        
        # Perform feature engineering
        taxi = feature_engineering(taxi_processed_obj, feature_cols=numeric_columns, date_cols=date_columns, discrete_cols=discrete_columns)

        # Create the processed data directory if it doesn't exist, and save the processed data
        os.makedirs(os.path.dirname(processed_data), exist_ok=True)
        taxi.to_csv(processed_data, index=None)
        log_message(logger, 'info', f'Preprocessing completed successfully')  # Log success
    except Exception as e:
        log_message(logger, 'error', f'Pipeline failed with error: {e}')  # Log error if the pipeline fails
        raise


if __name__ == '__main__':
    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser(prog='Preprocessing Steps')

    # Add arguments for the config file, raw dataset, and processed dataset paths
    parser.add_argument('--config_file', type=str, default=f'{config_}/params.yaml', help='Path to Yaml File')
    parser.add_argument('--raw_dataset', type=str, default=f'{raw_dir}/yellow_tripdata_2023-01.parquet', help='Path to raw dataset')
    parser.add_argument('--processed_dataset', type=str, default=f'{processed_dir}/taxi.csv', help='Path to processed folder')

    # Parse the arguments and run the main function
    args = parser.parse_args()
    main(args.config_file, args.raw_dataset, args.processed_dataset)
