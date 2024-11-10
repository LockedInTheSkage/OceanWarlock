import pandas as pd
import numpy as np
import re
from datetime import timedelta
from collections import deque
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

import loadBar
from csv_parser import CSVParser
from globals import RESOURCE_FOLDER, STEPSIZES, OUTPUT_WINDOW, INPUT_WINDOW, OUTPUT_FORECAST, DELETEABLE_COLUMNS, ONE_HOT_COLUMNS
from feature_engineer import FeatureEngineer
from exploring_data_functions import *


parser = CSVParser(RESOURCE_FOLDER)
total_df = parser.retrieve_training_data()

total_df["time"] = pd.to_datetime(total_df['time'])


def map_indexes_within_timewindow(df, time_col, timewindow):

    index_map = {}    

    within_window = deque()
    

    for current_idx in range(len((df))):
        current_row = df.iloc[current_idx]
        current_time = current_row[time_col]
        

        while within_window and (current_time - within_window[0][1] > timewindow):
            index_map[within_window[0][0]] = within_window[-1][0]
            within_window.popleft()
        
        if current_idx == df.index[-1]:
            for idx, _ in within_window:
                index_map[idx] = current_idx
        

        within_window.append((current_idx, current_time))
    
    return index_map


def calculate_time_diffs_within_window(df, time_deltas=[]):
    result_list= []
    vessel_dfs = df.groupby('vesselId')
    j=0
    for _, vessel_df in vessel_dfs:
        j+=1
        loadBar.load_bar(len(vessel_dfs), j)
        
        vessel_df = vessel_df.sort_values(by='time')
        groups = vessel_df.groupby(pd.Grouper(key="time", freq='5D'))
        
        for _, group in groups:
                group_deltas=time_deltas.copy()
                if len(group) < 2:
                    continue
                while group_deltas and group.iloc[-1]['time']-group.iloc[0]['time'] < group_deltas[0]:
                    group_deltas.pop(0)
                feature_rows=[group.iloc[0]]

                for i in range(len(group)):
                    future_row = group.iloc[i]

                    for current_row in feature_rows:
                        new_row = current_row.copy()
                        new_row['next_latitude'] = future_row['latitude']
                        new_row['next_longitude'] = future_row['longitude']
                        new_row['time_diff'] = (future_row['time'] - current_row['time']).total_seconds()
                        result_list.append(new_row)

                    if group_deltas:
                        if group.iloc[-1]['time']-future_row['time'] < group_deltas[0]:
                            group_deltas.pop(0)
                            feature_rows.append(future_row)
                        
    print("Concating")
    result_df = pd.DataFrame(result_list)

    last_rows = df.groupby('vesselId').tail(1)

    last_rows = last_rows.reset_index(drop=True)
    result_df
    
    return result_df, last_rows


timedeltas=[timedelta(days=3),timedelta(days=1)]

total_df, last_rows = calculate_time_diffs_within_window(total_df, timedeltas) #1 min 14 sec on no delta, 2 min 30 sec on 1 and 3 day delta

total_df.to_csv(RESOURCE_FOLDER+"/time_diff_training_data.csv", index=False)
last_rows .to_csv(RESOURCE_FOLDER+"/time_diff_training_st.csv", index=False)