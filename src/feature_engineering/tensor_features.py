import pandas as pd
import numpy as np
from loadBar import load_bar
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

lat_time_dif=10**5

def develop_features(df, derivative_locations=False):
    
    columns_to_drop = ['vesselId', 'portId', 'group_id', 'time_key']
    columns_to_tokenize = []
    columns_to_categorize = ['ISO', 'UN_LOCODE']


    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    print("Developing features with the following columns: ", df.columns)
    
    time_columns = [col for col in df.columns if col.startswith('time_')]
    time_columns.append('etaParsed')

    
    print("Categorizing...")
    df = categorize_columns(df, columns_to_categorize)
    print("Tokenizing...")
    df = tokenize_columns(df, columns_to_tokenize)
    print("Normalizing timestamps...")
    df = normalize_time(df, time_columns) 

    
    if not derivative_locations:
        return df
    
    print("Derivating locations...")

    time_cols = [col for col in df.columns if col.startswith('time')]
    latitude_cols = [col for col in df.columns if col.startswith('latitude')]
    longitude_cols = [col for col in df.columns if col.startswith('longitude')]
    time_cols.remove("time_0")
    # Create a copy of the DataFrame to avoid modifying the original
    df_transformed = df.copy()

    # Loop over the range of time, latitude, and longitude columns
    for i in range(1, len(time_cols)):
        # Calculate time difference
        time_diff = df[time_cols[i]] - df[time_cols[i-1]]
        # Replace time column with the time difference
        df_transformed[time_cols[i]] = time_diff.apply(float)

        # Calculate latitude and longitude differences and normalize by time difference
        lat_diff = df[latitude_cols[i]] - df[latitude_cols[i-1]]
        long_diff = df[longitude_cols[i]] - df[longitude_cols[i-1]]
        
        # Replace with the differences divided by time difference (avoid division by zero)
        df_transformed[latitude_cols[i]] = lat_time_dif*lat_diff / time_diff.replace(0.0, 10**24)  # Replace 0 with NA to avoid division by zero
        df_transformed[longitude_cols[i]] = lat_time_dif*long_diff / time_diff.replace(0.0, 10**24)

    return df_transformed

def normalize_time(df, time_columns):
    for time_col in time_columns:
        df[time_col] = df[time_col].apply(normalize_time_to_seconds)
    nomralize(df)
    return df

def normalize_time_to_seconds(time_str):
    base_date = datetime(2024, 1, 1)
    #last_date = datetime(2025, 12, 31)
    if pd.isna(time_str):  # Handle missing values
        return np.nan
    time_obj = pd.to_datetime(time_str)  # Convert to datetime object
    seconds_since_base = (time_obj - base_date).total_seconds()  # Seconds since base_date
    # max_seconds_range = (last_date - base_date).total_seconds()  # Range from base_date to last_date
    return seconds_since_base # / max_seconds_range  # Normalize to range 0 to 1

def floating_conv(timestamp):
    """
    Input:
    timestamp: A pandas Timestamp object
    
    Output:
    seconds: The number of seconds since the (January 1, 2024)
    """
    if not (isinstance(timestamp, pd.Timestamp) or isinstance(timestamp, float) or isinstance(timestamp, int)):
        return -1.0
    elif isinstance(timestamp, pd.Timestamp):
        return float((timestamp - pd.Timestamp('2024-01-01')).total_seconds())/3600
    return float(timestamp)/3600

def tokenize_columns(df, columns):

    df= pd.get_dummies(df, columns=columns, drop_first=True)
    return df

def categorize_columns(df, columns):
    for col in columns:
        if not col in df.columns: 
            continue
        df[col], _ = pd.factorize(df[col])
    return df

def nomralize(df):
    
    return (df - df.min()) / (df.max() - df.min())
