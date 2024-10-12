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
        return parsed_data
    
    
    def load_train_and_test_data(self, df_dict):
        if self.print_progress:
            print("Sorting into tests and training data")
        keys = list(df_dict.keys())
        train_size = int(self.train_test_split * len(keys))

        training_keys = keys[:train_size]
        test_keys = keys[train_size:]

        training_dict={}
        test_dict={}

        for i, key in enumerate(training_keys):
            if self.print_progress:
                load_bar(len(keys), i+1)
            training_dict[key] = df_dict[key]
        for i, key in enumerate(test_keys):
            if self.print_progress:
                load_bar(len(keys), i+len(training_keys))
            test_dict[key] = df_dict[key]
        
        return training_dict, test_dict

    def load_training_data(self, data):
        training_dict, test_dict = self.load_train_and_test_data(data)
        if self.print_progress:
            print("Formating training data")
        x,y=self.format_dictionaries(training_dict)
        if self.print_progress:
            print("Formating test data")
        x_t,y_t=self.format_dictionaries(test_dict)
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

    