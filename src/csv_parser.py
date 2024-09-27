import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def parse_time(raw_time):
    try:
    
        # Define the format without the year
        date_format = "%m-%d %H:%M"
    
        # Parse the cleaned string into a datetime object
        parsed_datetime = datetime.strptime(raw_time, date_format)

        # Add placeholder year 2024.

        return parsed_datetime.replace(year = 2024)
    
    except ValueError:
        return None

def retrieve_ais(path):
    df_ais = pd.read_csv('../resources/ais_train.csv', sep='|')
    df_ais.head()
    df_ais['time'] = pd.to_datetime(df_ais['time'])
    df_ais['etaParsed'] = df_ais['etaRaw'].apply(parse_time)
    df_ais.drop(['etaRaw'], axis = 1)

    df_ais.isna().sum() #Estimated arrival time (1615 entries) and portId (1615 entries)

    df_ais.loc[df_ais.duplicated()] #No duplicated rows

    return df_ais

def retrieve_all_data():
    
    
    