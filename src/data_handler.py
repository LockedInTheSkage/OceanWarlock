import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from csv_parser import CSVParser
from path_finder import path_sorter
from loadBar import load_bar
from sklearn.model_selection import train_test_split

target_columns = ["latitude_0", "longitude_0"] 
unrelated_columns = ["cog_0", "sog_0"]

class LocalToLargeDataLoader():
    
    def __init__(
        self,  train_test_ratio: float = 0.2, print_progress: bool = False
    ):
        self.train_test_split = train_test_ratio
        self.print_progress = print_progress
        self.eval_data = ["latitude", "longitude"]

    def load_raw_data(self, path=None):
        if path:
            parser = CSVParser(path)
        else:
            parser = CSVParser()
        if self.print_progress:
            print("Retrieving training data...")
        parsed_data = parser.retrieve_training_data()
        return parsed_data
    

    def load_training_data(self, data):
        if self.print_progress:
            print("Sorting into tests and training data")
        unrelated_columns.extend(target_columns)
        x = data.drop(unrelated_columns, axis=1)  # Features (all columns except 'target_column')
        y = data[target_columns]  # Labels (the target column)
        x, x_t, y, y_t = train_test_split(x, y, test_size=self.train_test_split, random_state=np.random.randint(0, 1000))
        return x, y, x_t, y_t
    
    def format_dictionaries(self, input_dict):
        x={}
        y={}
        for i, key in enumerate(input_dict.keys()):
            if self.print_progress:
                load_bar(len(input_dict.keys()), i+1)
            input_dict[key][1]['time']=input_dict[key][0].iloc[0:1][['time']].values[0][0]
            x[key] = [input_dict[key][0].iloc[1:], input_dict[key][1]]
            y[key] = input_dict[key][0].iloc[0:1][['latitude', 'longitude']]
        return x, y

    