import os
import pandas as pd
from datetime import datetime
from typing import Tuple, List

ROOT = './data/sentinel_data/fields/'

def load_weather_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

def get_current_datetime() -> datetime.date:
    return datetime.now().date()

def get_weather_forecast(field_id: int, root: str = ROOT, days: int = 5) -> Tuple[List[str], List[float], List[float]]:
    filepath = os.path.join(root, str(field_id), 'weather', 'weather.csv')
    df = load_weather_data(filepath)
    
    date = get_current_datetime()

    df['validTimeUtc'] = pd.to_datetime(df['validTimeUtc'])

    cond_1 = (df['validTimeUtc'].dt.day > date.day)
    cond_2 = (df['validTimeUtc'].dt.month > date.month)
    cond_3 = (df['validTimeUtc'].dt.month == date.month)

    df = df.loc[((~cond_1 & cond_2) | (cond_1 & cond_3))]

    sample = df.iloc[:days]
    sample['validTimeUtc'] = sample['validTimeUtc'].astype(str)

    date_time = sample['validTimeUtc'].tolist()
    temperature = sample['Temperature'].tolist()
    precipitation = sample['Precipitation'].tolist()

    return date_time, temperature, precipitation
