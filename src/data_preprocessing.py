import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def transform_target(y):
  return np.log1p(y).rename('log_'+y.name)


def preprocess_data(df):
     # Convert pickup and dropoff timestamps
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

    # Add new features
    df['weekday'] = df['pickup_datetime'].dt.weekday
    df['month'] = df['pickup_datetime'].dt.month
    df['hour'] = df['pickup_datetime'].dt.hour
    # Save preprocessed dataframe to CSV
    df.to_csv('data/raw/preprocessed_data.csv', index=False)

    return df

def split_data(df):
    # Split the dataset into training and testing.
    X = df[['hour', 'weekday', 'month']]  # Add more features later
    y = df['trip_duration']
    y = transform_target(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)
