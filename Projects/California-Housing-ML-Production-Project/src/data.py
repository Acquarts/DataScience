import pandas as pd

path = 'data/housing.cvs'

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

