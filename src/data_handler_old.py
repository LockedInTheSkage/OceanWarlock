
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


TRAINING_DATA_FILE = "../data/chessData.csv"
TARGET_FEATURE = "Evaluation"

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
        self, cursor: int = 0, chunksize: int = 1_000_000, train_test_split: float = 0.8
    ):
        self.cursor = cursor
        self.chunksize = chunksize
        self.train_test_split = train_test_split

    def _load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = pd.read_csv(TRAINING_DATA_FILE)
        # Load a chunk of the data
        data = data.iloc[self.cursor : self.cursor + self.chunksize]

        # Split the data into training and testing sets
        train_size = int(self.train_test_split * len(data))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        return train_data, test_data

    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        data, _ = self._load_raw_data()
        target_column = TARGET_FEATURE
        x = data.drop(target_column, axis=1)
        y = data[target_column]
        # Make sure the x and y are DataFrames
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        return x, y

    def load_raw_test_data(self) -> pd.DataFrame:
        _, data = self._load_raw_data()
        x = data.drop(TARGET_FEATURE, axis=1)
        y = data[TARGET_FEATURE]
        # Make sure the x and y are DataFrames
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        return x, y


