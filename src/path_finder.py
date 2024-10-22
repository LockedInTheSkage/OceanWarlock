import pandas as pd
from loadBar import load_bar
import time

time_series_columns=['time', 'cog', 'sog', 'latitude', 'longitude']

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
    df: DataFrame object 

    Output:
    A dictionary where each key is a tuple ('vesselId', 'time'), and the value is a list of sorted entries (based on 'time') for that combination.
    """
    bucket_list = []
    WINDOW_SIZE = 10
    
    
    grouped = df.groupby(['vesselId'])
    i=0
    timedif=0
    for group_id, group_df in grouped:
        load_bar(len(grouped), i+1)
        i+=1

        sorted_group = group_df.sort_values(by='time', ascending=False)
        for cursor in range(len(group_df)//WINDOW_SIZE+1):
            if(cursor*WINDOW_SIZE==len(group_df)):
                break

            window = sorted_group.loc[sorted_group.index[cursor * WINDOW_SIZE]:sorted_group.index[min((1 + cursor) * WINDOW_SIZE, len(sorted_group) - 1) - 1]]

            filtered_columns = df.columns[~df.columns.isin(time_series_columns)].tolist()

            if len(window) < WINDOW_SIZE:
                padding = pd.DataFrame(-1, index=range(WINDOW_SIZE - len(window)), columns=df.columns)
                window = pd.concat([window, padding], ignore_index=True)

            details = window[filtered_columns].loc[window.index[1]]
            bucketvalues = window[time_series_columns] 

            details_df = pd.DataFrame([details]).reset_index(drop=True)

            flattened_bucketvalues = bucketvalues.values.flatten()

            bucket_columns = [f'{time_series_columns[i%len(time_series_columns)]}_{i//len(time_series_columns)}' for i in range(len(flattened_bucketvalues))]
            bucketvalues_df = pd.DataFrame([flattened_bucketvalues], columns=bucket_columns)
            final_df = pd.concat([details_df, bucketvalues_df], axis=1)
            bucket_list.append(final_df)

    print("")
    print("Concatting dataframes")
    print(f"Number of dataframes: {len(bucket_list)}")
    combined_df = pd.concat(bucket_list, ignore_index=True)
    
    return combined_df
