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
        self.df[new_feature.name] = new_feature

    def get_dataframe(self):
        return self.df

# Example usage:
def example_feature(df):
    new_feature = df['existing_column'] * 2
    new_feature.name = 'new_feature'
    return new_feature

df = pd.DataFrame({'existing_column': [1, 2, 3]})
fe = FeatureEngineer(df)
fe.add_feature(example_feature)
new_df = fe.get_dataframe()
print(new_df)