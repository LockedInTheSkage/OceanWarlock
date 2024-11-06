import pandas as pd
from loadBar import load_bar
import time

time_series_columns=['time', 'cog', 'sog', 'latitude', 'longitude']
WINDOW_SIZE = 10

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

def test_path_sorter(df, test_id_df):
    """
    Input:
    df: DataFrame object 

    Output:
    A dictionary where each key is a tuple ('vesselId', 'time'), and the value is a list of sorted entries (based on 'time') for that combination.
    """
    bucket_list = []
    print("New")
    

    
    df_relevant = df[df['vesselId'].isin(test_id_df['vesselId'])]
    grouped = df_relevant.groupby(['vesselId'])
    i=0


    for group_id, group_df in grouped:
        load_bar(len(grouped), i+1)
        i+=1

        sorted_group = group_df.sort_values(by='time', ascending=False)

        window = sorted_group.loc[sorted_group.index[0]:sorted_group.index[min((WINDOW_SIZE - 1), len(sorted_group) - 1) - 1]]

        filtered_columns = df.columns[~df.columns.isin(time_series_columns)].tolist()

        if len(window) < WINDOW_SIZE:
            padding = pd.DataFrame(-1, index=range(WINDOW_SIZE-1 - len(window)), columns=df.columns)
            window = pd.concat([window, padding], ignore_index=True)

        details = window[filtered_columns].loc[window.index[1]]
        bucketvalues = window[time_series_columns] 

        details_df = pd.DataFrame([details]).reset_index(drop=True)

        flattened_bucketvalues = bucketvalues.values.flatten()

        bucket_columns = [f'{time_series_columns[i%len(time_series_columns)]}_{i//len(time_series_columns)+1}' for i in range(len(flattened_bucketvalues))]
        bucketvalues_df = pd.DataFrame([flattened_bucketvalues], columns=bucket_columns)
        final_df = pd.concat([details_df, bucketvalues_df], axis=1)

        query_df=test_id_df.query(f'vesselId == "{group_id[0]}"').drop(columns=['vesselId']).reset_index(drop=True)

        result_df = pd.DataFrame()

        if len(final_df) == 1:
            repeated_final_df = pd.concat([final_df]*len(query_df), ignore_index=True)
            
            result_df = pd.concat([query_df.reset_index(drop=True), repeated_final_df], axis=1)
        else:
            print("Error: final_df should contain only one row.")
        result_df['time_0']=result_df['time']
        result_df=result_df.drop(columns=['time', 'scaling_factor'])
        bucket_list.append(result_df)
    print("")
    print("Concatting dataframes")
    print(f"Number of dataframes: {len(bucket_list)}")
    combined_df = pd.concat(bucket_list, ignore_index=True)
    combined_df = combined_df.sort_values(by='ID')

    return combined_df

