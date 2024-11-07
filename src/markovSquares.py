import pandas as pd
import numpy as np
import loadBar
from globals import RESOURCE_FOLDER, MARKOV_SIZE

class MarkovSquares():

    def __init__(self, square_size):
        self.square_size = square_size
        self.normalized_markov_matrix = None
        self.direction_columns = ["SW", "S", "SE", "W", "C", "E", "NW", "N", "NE"]

    def markovSquares(self, df, sorting_column):
        unique_sorts = df[sorting_column].unique()

        markov_matrix = np.zeros((180//self.square_size,360//self.square_size,9))
        print("Generating Markov Matrix")
        for j, sorts in enumerate(unique_sorts):
            
            loadBar.load_bar(len(unique_sorts),j)
            sort_df = df[df[sorting_column] == sorts]

            #Iterate through all columns for input features
            for i in range(len(sort_df)-1):
                entry1 = sort_df.iloc[i]
                entry2 = sort_df.iloc[i+1]

                lat1 = entry1["latitude"]
                lon1 = entry1["longitude"]

                lat2 = entry2["latitude"]
                lon2 = entry2["longitude"]

                lat_diff = (lat1-lat2)//self.square_size
                lon_diff = (lon1-lon2)//self.square_size

                inner_idx=self.markov_index_inner(lat_diff, lon_diff)

                lat_idx=int(lat1//self.square_size)
                lon_idx=int(lon1//self.square_size)
            
                markov_matrix[lat_idx][lon_idx][inner_idx]+=1
            
        self.normalized_markov_matrix  = markov_matrix / np.sum(markov_matrix, axis=-1, keepdims=True)

    def add_as_columns(self, df, lat_col="latitude", lon_col="longitude"):
        # Define column names for the new data
        column_names = self.direction_columns
        
        # Initialize a DataFrame to hold the results
        results_df = pd.DataFrame(columns=column_names, index=df.index)
        
        print("Adding Markov Squares as columns")
        i=0
        for idx, row in df.iterrows():
            i+=1
            loadBar.load_bar(len(df),i)
            # Extract latitude and longitude values
            latitude = row[lat_col]
            longitude = row[lon_col]
            
            # Check for NaN values in latitude and longitude
            if pd.notna(latitude) and pd.notna(longitude):
                # Get the processed values (assumes `get_markov_square` returns a list of 9 values)
                processed_values = self.get_markov_square(latitude, longitude)
                # Set the result in the corresponding row of results_df
                results_df.loc[idx] = processed_values
            else:
                # Set NaN for each new column if latitude or longitude is NaN
                results_df.loc[idx] = [np.nan] * 9

        # Concatenate results_df with the original df along the columns axis
        df = pd.concat([df, results_df], axis=1)
        df= df[column_names]
        return df


    def get_markov_square(self, lat, lon):
        lat_idx = int(lat//self.square_size)
        lon_idx = int(lon//self.square_size)
        return self.normalized_markov_matrix[lat_idx][lon_idx]

    def markov_index_inner(self, lat_diff, lon_diff):
        lat_index = 0 if lat_diff <= -1 else (1 if lat_diff == 0 else 2)
        lon_index = 0 if lon_diff <= -1 else (1 if lon_diff == 0 else 2)
        index = lat_index * 3 + lon_index
        return index
    
    

def apply_markov(df):
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return None
    markov = MarkovSquares(MARKOV_SIZE)
    markov.normalized_markov_matrix = np.load(RESOURCE_FOLDER+"/markov_matrix.npy")
    df_m = markov.add_as_columns(df, lat_col="latitude", lon_col="longitude")
    df_m = df_m[markov.direction_columns]
    df_m.name = markov.direction_columns
    return df_m