import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from csv_parser import CSVParser
from path_finder import path_sorter
from loadBar import load_bar

class DataLoader(ABC):
    @abstractmethod
    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the data and return the features and target variable."""
        pass

    @abstractmethod
    def load_raw_test_data(self) -> pd.DataFrame:
        """Load the test data."""
        pass


class LocalToLargeDataLoader(DataLoader):
    
    def __init__(
        self, cursor: int = 0, chunksize: int = 1_000_000, train_test_split: float = 0.8, print_progress: bool = False
    ):
        self.cursor = cursor
        self.chunksize = chunksize
        self.train_test_split = train_test_split
        self.print_progress = print_progress
        self.eval_data = ["latitude", "longitude"]

    def _load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        parser = CSVParser()
        if self.print_progress:
            print("Retrieving training data...")
        parsed_data = parser.retrieve_training_data()
        df_paths = path_sorter(parsed_data)
        # Load a chunk of the data
        keys = list(df_paths.keys())[self.cursor : self.cursor + self.chunksize]

        # Split the data into training and testing sets
        train_size = int(self.train_test_split * len(keys))
        training_keys = keys[:train_size]
        test_keys = keys[train_size:]
        return df_paths, training_keys, test_keys

    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_dict, keys, _ = self._load_raw_data()
        print(keys[0])
        print(df_dict[keys[0]])
        x = keys.drop(self.eval_data, axis=1)
        y=keys[self.eval_data]
        # Make sure the x and y are DataFrames
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        return x, y

    def load_raw_test_data(self) -> pd.DataFrame:
        df_dict, _, keys = self._load_raw_data()
        x = keys.drop(self.eval_data, axis=1)
        y=keys[self.eval_data]
        # Make sure the x and y are DataFrames
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        return x, y
