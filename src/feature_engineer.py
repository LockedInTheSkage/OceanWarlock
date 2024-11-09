import pandas as pd



class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def add_feature(self, func, *args, **kwargs):
        """
        Apply a function to the dataframe to add a new feature.
        
        Parameters:
        func (callable): A function that takes a dataframe and returns a series.
        *args: Additional positional arguments to pass to the function.
        **kwargs: Additional keyword arguments to pass to the function.
        """
        new_feature = func(self.df, *args, **kwargs)
        if new_feature is not None:
            if isinstance(new_feature, pd.DataFrame):  # Handle DataFrame case
                self.df = pd.concat([self.df, new_feature], axis=1)
            else:
                self.df[new_feature.name] = new_feature

    def get_dataframe(self):
        return self.df
    
    def apply_features(self, features):
        """
        Apply a list of functions to the dataframe to add new features.
        
        Parameters:
        features (list): A list of functions that take a dataframe and return a series.
        """
        for func in features:
            self.add_feature(func)