import yaml
import pandas as pd 
from log_file import set_up_logs, log_message


logger = set_up_logs(__name__, 'utils.log') 

def load_config(path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        path (str): The path to the configuration file.

    Returns:
        dict: The loaded configuration as a dictionary, or None if there was an error.
    """
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
            log_message(logger, 'info', f'Parameter file loaded: {path}')
            return params
    except FileNotFoundError:
        log_message(logger, 'error', f'File not found: {path}')
    except Exception as e:
        log_message(logger, 'error', f'Error opening file {path}: {str(e)}')
    return None  # Return None to indicate failure

def load_data(file: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load data from a parquet file and split it into features (X) and target (y).

    Args:
        file (str): The path to the parquet file.
        target (str): The name of the target column.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the feature DataFrame (X) and target Series (y).
    
    Raises:
        ValueError: If the target column is not found in the dataset.
    """
    try:
        data = pd.read_parquet(file)
        
        # Ensure the target column exists in the DataFrame
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in the dataset.")
        
        # Split the DataFrame into features (X) and target (y)
        X = data.drop(columns=target)
        y = data[target]
        
        return X, y
    except Exception as e:
        log_message(logger, 'error', f'Error loading data from {file}: {str(e)}')
        raise  # Re-raise the exception to let the caller handle it
