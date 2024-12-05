import yaml
import pandas as pd 
import logging




def load_config(path: str):
    with open(path, 'r') as file:
        yaml.safe_load(file)


def load_data(file):
    
    data = pd.read_csv(file)
