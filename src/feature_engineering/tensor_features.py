import pandas as pd
import numpy as np
from loadBar import load_bar
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

def develop_features(df):
    
    columns_to_drop = ['vesselId', 'portId', 'group_id', 'time_key']
    columns_to_tokenize = ['UN_LOCODE']
    columns_to_categorize = ['ISO']


    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    print("Developing features with the following columns: ", df.columns)
    
    time_columns = [col for col in df.columns if col.startswith('time_')]
    time_columns.append('etaParsed')

    df = tokenize_columns(df, columns_to_tokenize)
    df = categorize_columns(df, columns_to_categorize)
    df = normalize_time(df, time_columns)  
    
    return df

def normalize_time(df, time_columns):
    for time_col in time_columns:
        df[time_col] = df[time_col].apply(normalize_time_to_seconds)
    return df

def normalize_time_to_seconds(time_str):
    base_date = datetime(2023, 1, 1)
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
    seconds: The number of seconds since the (January 1, 2023)
    """
    if not (isinstance(timestamp, pd.Timestamp) or isinstance(timestamp, float) or isinstance(timestamp, int)):
        return -1.0
    elif isinstance(timestamp, pd.Timestamp):
        return float((timestamp - pd.Timestamp('2023-01-01')).total_seconds())
    return float(timestamp)

def tokenize_columns(df, columns):
    tokenizer = Tokenizer()
    for col in columns:
        if not col in df.columns: 
            continue
        tokenizer.fit_on_texts(df[col]) 
        df[col] = tokenizer.texts_to_sequences(df[col])
    return df

def categorize_columns(df, columns):
    for col in columns:
        if not col in df.columns: 
            continue
        df[col], _ = pd.factorize(df[col])
    return df