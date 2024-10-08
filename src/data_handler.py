import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from csv_parser import CSVParser
from path_finder import path_sorter
from loadBar import load_bar



class LocalToLargeDataLoader():
    
    def __init__(
        self,  train_test_split: float = 0.8, print_progress: bool = False
    ):
        self.train_test_split = train_test_split
        self.print_progress = print_progress
        self.eval_data = ["latitude", "longitude"]

    def load_raw_data(self):
        parser = CSVParser()
        if self.print_progress:
            print("Retrieving training data...")
        parsed_data = parser.retrieve_training_data()
        df_paths = path_sorter(parsed_data)
        return df_paths
    
    def _load_train_and_test_data(self) -> tuple[dict, dict]:
        df_dict = self.load_raw_data()

        filtered_dict = {k: v for k, v in df_dict.items() if len(v) > 1}
        keys = list(filtered_dict.keys())
        train_size = int(self.train_test_split * len(keys))

        training_keys = keys[:train_size]
        test_keys = keys[train_size:]

        training_dict = {k: v for k, v in filtered_dict.items() if k in training_keys}
        test_dict = {k: v for k, v in filtered_dict.items() if k in test_keys}
        
        return training_dict, test_dict

    def load_raw_training_data(self):
        training_dict, test_dict = self._load_train_and_test_data()
        x,y=self.format_dictionaries(training_dict)
        x_t,y_t=self.format_dictionaries(test_dict)
        return x, y, x_t, y_t
    
    def format_dictionaries(self, input_dict):
        x={}
        y={}
        for key in input_dict.keys():
            input_dict[key][1]['time']=input_dict[key][0].iloc[0:1][['time']].values[0][0]
            x[key] = [input_dict[key][0].iloc[1:], input_dict[key][1]]
            y[key] = input_dict[key][0].iloc[0:1][['latitude', 'longitude']]
        return x, y

    