
def path_finder(df, id, eta):
    """
    Input:
    df: dataFrame object
    id: String. Id of vessel.
    eta: DateTime data. Estimated arrival time

    Output:
    df_paths: dataFrame with times and corresponding latitudes and longitudes sorted for given id and eta.
    """
    df_paths = df[(df["vesselId"]==id) & (df["etaParsed"]==eta)][["time","latitude","longitude"]].copy()
    return df_paths

