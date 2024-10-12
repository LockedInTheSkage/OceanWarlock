import pandas as pd
import numpy as np
from loadBar import load_bar

def develop_features(x: dict, y:dict):
    """
    Input:
    x: A dictionary with keys as vesselId and values as a list of two dataframes. 
       The first dataframe contains the time series data, and the second dataframe contains the details.
    y: A dictionary with keys as vesselId and values as a dataframe with the target variable
    
    Output:
    array_x: A numpy array with features developed from the input dataframes.
    array_y: A numpy array with the target values.
    """
    WINDOW_SIZE = 10

    array_x = []
    array_y = []
    bo=True
    for i, key in enumerate(x.keys()):
        load_bar(len(x.keys()), i)
        # Extract the first dataframe (time series data) and second dataframe (spatial details)
        df_timeseries = x[key][0].copy()
        df_details = pd.DataFrame(x[key][1].copy())
        df_y= y[key]

        
        # Preprocessing on the first dataframe
        # Drop unwanted columns 'vesselId' and 'portId'
        df_details = df_details.drop(['vesselId', 'portId'])
        df_details = df_details.map(floating_conv)
        # Convert the dataframe to a one-dimensional feature vector
        feature_vector = df_details.to_numpy().flatten().tolist()
        
        feature_vector = np.array(feature_vector)
        # Extract the first WINDOW_SIZE rows from the second dataframe (df_details)

        # Convert the details dataframe into a flat array and concatenate with the first dataframe's feature vector
        df_timeseries = df_timeseries.map(floating_conv)
        timeseries_vector = df_timeseries.to_numpy().flatten().tolist()
        column_n = len(df_timeseries.columns.tolist())
        timestamps=timeseries_vector[::column_n]
        
        

        timeseries_vector = np.array(timeseries_vector)

        combined_features = np.concatenate([feature_vector, timeseries_vector])
        # Append the combined features to array_x
        array_x.append(combined_features)

        # Append the target to array_y
        array_y.append(df_y[['latitude', 'longitude']].to_numpy().flatten())

    # Convert lists to numpy arrays for easier handling in tensors
    array_x = np.array(array_x)
    array_y = np.array(array_y)

    print("[====================] 100% complete")
    return array_x, array_y

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