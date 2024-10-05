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
    df_ais = pd.read_csv(path, sep='|')
    df_ais['time'] = pd.to_datetime(df_ais['time'])
    df_ais['etaParsed'] = df_ais['etaRaw'].apply(parse_time)
    df_ais=df_ais.drop(['etaRaw'], axis = 1)
    return df_ais

def retrieve_ports(path):
    df_ports = pd.read_csv(path, sep='|')
    df_ports['portLongitude'] = df_ports['longitude']
    df_ports['portLatitude'] = df_ports['latitude']
    df_ports=df_ports.drop(['name', 'portLocation', 'countryName', 'longitude', 'latitude'], axis = 1)
    return df_ports

def retrieve_schedules(path):
    df_schedules = pd.read_csv(path, sep='|')
    df_schedules['sailingDate'] = pd.to_datetime(df_schedules['sailingDate'])
    df_schedules['arrivalDate'] = pd.to_datetime(df_schedules['arrivalDate'])

    df_schedules=df_schedules.drop(['shippingLineName', 'portName', 'portLongitude', 'portLatitude'], axis = 1)
    print(df_schedules.head())
    return df_schedules

def retrieve_vessels(path):
    df_vessels = pd.read_csv(path, sep='|')
    return df_vessels

def retrieve_tests(path):
    df_tests = pd.read_csv(path, sep='|')
    df_tests['time'] = pd.to_datetime(df_tests['time'])
    return df_tests

def retrieve_all_data():
    folderpath = 'resources/'
    df_ais=retrieve_ais(folderpath+'ais_train.csv')
    df_ports=retrieve_ports(folderpath+'ports.csv')
    df_schedules=retrieve_schedules(folderpath+'schedules_to_may_2024.csv')
    df_vessels=retrieve_vessels(folderpath+'vessels.csv')
    df_tests=retrieve_tests(folderpath+'ais_test.csv')

    
    result = pd.merge(df_ais, df_ports, on='portId')
    result = pd.merge(result, df_vessels, on='vessel_id')
    result = pd.merge(result, df_schedules, on='portId')
    result = result.drop(['portId'], axis = 1)
    print(result.head())


if __name__ == '__main__':
    retrieve_all_data()
    