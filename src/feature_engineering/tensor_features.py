import pandas as pd
import numpy as np

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
    for key in x.keys():
        # Extract the first dataframe (time series data) and second dataframe (spatial details)
        df_timeseries = x[key][0].copy()
        df_details = pd.DataFrame(x[key][1].copy())
        df_y= y[key]


        # Preprocessing on the first dataframe
        # Drop unwanted columns 'vesselId' and 'portId'
        df_details = df_details.drop(['vesselId', 'portId'])

        

        # Convert the dataframe to a one-dimensional feature vector
        feature_vector = df_details.to_numpy().flatten()

        # Extract the first WINDOW_SIZE rows from the second dataframe (df_details)
        if len(df_timeseries) < WINDOW_SIZE:
            df_timeseries_window = df_timeseries
        else:
            df_timeseries_window = df_timeseries.head(WINDOW_SIZE)

        # Convert the details dataframe into a flat array and concatenate with the first dataframe's feature vector
        timeseries_vector = df_timeseries_window.to_numpy().flatten()
        if len(timeseries_vector) < WINDOW_SIZE * len(df_timeseries.columns):
            timeseries_vector = np.concatenate([timeseries_vector, np.zeros(WINDOW_SIZE * len(df_timeseries.columns) - len(timeseries_vector))])
        combined_features = np.concatenate([feature_vector, timeseries_vector])
        # Append the combined features to array_x
        array_x.append(combined_features)

        # Append the target to array_y
        array_y.append(df_y[['latitude', 'longitude']].to_numpy().flatten())

    # Convert lists to numpy arrays for easier handling in tensors
    array_x = np.array(array_x)
    array_y = np.array(array_y)

    return array_x, array_y

def convert_timestamp_to_seconds(timestamp):
    """
    Input:
    timestamp: A pandas Timestamp object
    
    Output:
    seconds: The number of seconds since the (January 1, 2023)
    """
    if isinstance(timestamp, pd.Timestamp):
        return float((timestamp - pd.Timestamp('2023-01-01')).total_seconds())
    return float(timestamp)