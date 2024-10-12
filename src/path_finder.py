import pandas as pd
from loadBar import load_bar
import time

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
    df: DataFrame object 

    Output:
    A dictionary where each key is a tuple ('vesselId', 'time'), and the value is a list of sorted entries (based on 'time') for that combination.
    """
    bucket_dict = {}
    WINDOW_SIZE = 10
    
    
    grouped = df.groupby(['vesselId'])
    bo=60
    i=0
    timedif=0
    for group_id, group_df in grouped:
        load_bar(len(grouped), i+1)
        i+=1

        sorted_group = group_df.sort_values(by='time', ascending=False)
        for cursor in range(len(group_df)//WINDOW_SIZE+1):
            if(cursor*WINDOW_SIZE==len(group_df)):
                break
            window=sorted_group.loc[sorted_group.index[cursor*WINDOW_SIZE]:sorted_group.index[min((1+cursor)*WINDOW_SIZE,len(sorted_group)-1)-1]]
            

            filtered_columns = df.columns[~df.columns.isin(time_series_columns)].tolist()

            
            if len(window) < WINDOW_SIZE:
                padding = pd.DataFrame(-1, index=range(WINDOW_SIZE - len(window)), columns=df.columns)
                window= pd.concat([window, padding], ignore_index=True)

     
            details = window[filtered_columns].loc[window.index[1]]
            time_key=window['time'].loc[window.index[0]]


            bucketvalues=window[time_series_columns]
            bucket_dict[(group_id,time_key)] = [bucketvalues, details]

            
            
    
    return bucket_dict
