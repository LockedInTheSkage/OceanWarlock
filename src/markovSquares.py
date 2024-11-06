import pandas as pd
import numpy as np
import loadBar

class MarkovSquares():

    def __init__(self, square_size):
        self.square_size = square_size
        self.normalized_markov_matrix = None

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

    def add_as_columns(self, df):
        # Create lists to hold the new column data
        column_names = ["SW", "S", "SE", "W", "C", "E", "NW", "N", "NE"]
        precentage_lists = [[]]*9
        
        # Iterate through each row and apply the `process_coordinates` function
        for _, row in df.iterrows():
            # Extract latitude and longitude values
            latitude = row["latitude"]
            longitude = row["longitude"]
            
            # Check for NaN values and skip if necessary
            if pd.notna(latitude) and pd.notna(longitude):
                # Get the processed values
                processed_values = self.get_markov_square(latitude, longitude)
                
                # Append the processed values to the respective lists
                for i in range(9):
                    precentage_lists[i].append(processed_values[i])
            else:
                # If latitude or longitude is NaN, append NaN for each new column
                for i in range(9):
                    precentage_lists[i].append(np.nan)
        
        # Add the new columns to the DataFrame
        for i in range(9):
            df[column_names[i]] = precentage_lists[i]
        
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
    
    markov = MarkovSquares(2)
    markov.normalized_markov_matrix = np.load("../../resources/markov_matrix.npy")
    df = markov.add_as_columns(df)
    return df