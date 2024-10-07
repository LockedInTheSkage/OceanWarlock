
def path_finder(df, id, eta):
    """
    Input:
    df: dataFrame object
    id: String. Id of vessel.
    eta: DateTime data. Estimated arrival time

    Output:
    df_paths: dataFrame with times and corresponding latitudes and longitudes sorted for given vesselId and eta.
    """
    df_paths = df[(df["vesselId"]==id) & (df["etaParsed"]==eta)][["time","latitude","longitude"]].copy()
    return df_paths


def path_sorter(df):
    """
    Input:
    df: DataFrame object with columns ['vesselId', 'eta', 'time', 'latitude', 'longitude']

    Output:
    A dictionary where each key is a tuple ('vesselId', 'eta'), and the value is a list of sorted entries (based on 'time') for that combination.
    """
    # Create an empty dictionary to hold the buckets
    bucket_dict = {}

    # Group the DataFrame by 'vesselId' and 'eta'
    grouped = df.groupby(['vesselId', 'etaParsed'])

    # Iterate over each group
    for (group_id, group_eta), group_df in grouped:
        # Sort the group by 'time'
        sorted_group = group_df.sort_values(by='time')

        # Convert each row of sorted group to a list and store in the dictionary
        # Extract only the desired columns: 'time', 'latitude', 'longitude'
        bucket_dict[(group_id, group_eta)] = sorted_group[['time', 'latitude', 'longitude']].values.tolist()
    
    return bucket_dict
