import pandas as pd
time_series_columns=['time', 'latitude', 'longitude']

def path_finder(df, id, eta):
    """
    Input:
    df: dataFrame object
    id: String. Id of vessel.
    eta: DateTime data. Estimated arrival time

    Output:
    df_paths: dataFrame with times and corresponding latitudes and longitudes sorted for given vesselId and eta.
    """
    df_paths = df[(df["vesselId"]==id) & (df["etaParsed"]==eta)][time_series_columns].copy()
    return df_paths


def path_sorter(df):
    """
    Input:
    df: DataFrame object with columns ['vesselId', 'eta', 'time', 'latitude', 'longitude']

    Output:
    A dictionary where each key is a tuple ('vesselId', 'eta'), and the value is a list of sorted entries (based on 'time') for that combination.
    """
    bucket_dict = {}

    grouped = df.groupby(['vesselId'])
    bo=True
    for group_id, group_df in grouped:
        sorted_group = group_df.sort_values(by='time', ascending=False)

        filtered_columns = df.columns[~df.columns.isin(time_series_columns)].tolist()

        details = df[filtered_columns].loc[df.index[1]]

        bucketvalues=sorted_group[time_series_columns]
        bucket_dict[group_id] = [bucketvalues, details]
            
    
    return bucket_dict
