import pandas as pd
import numpy as np


def markovSquares(df, sorting_column, square_size):
    unique_sorts = df[sorting_column].unique()

    markov_matrix = np.zeros((180//square_size,360//square_size,9))

    for sorts in unique_sorts:
        sort_df = df[df[sorting_column] == sorts]

        #Iterate through all columns for input features
        for i in range(len(sort_df)-1):
            entry1 = sort_df.iloc[i]
            entry2 = sort_df.iloc[i+1]

            lat1 = entry1["latitude"]
            lon1 = entry1["longitude"]

            lat2 = entry2["latitude"]
            lon2 = entry2["longitude"]

            lat_diff = (lat1-lat2)//square_size
            lon_diff = (lon1-lon2)//square_size

            inner_idx=markov_index_inner(lat_diff, lon_diff)

            lat_idx=int(lat1//square_size)
            lon_idx=int(lon1//square_size)
         
            markov_matrix[lat_idx][lon_idx][inner_idx]+=1
        
    normalized_markov_matrix = markov_matrix / np.sum(markov_matrix, axis=-1, keepdims=True)

    return normalized_markov_matrix





def markov_index_inner(lat_diff, lon_diff):
    lat_index = 0 if lat_diff <= -1 else (1 if lat_diff == 0 else 2)
    lon_index = 0 if lon_diff <= -1 else (1 if lon_diff == 0 else 2)
    index = lat_index * 3 + lon_index
    return index
    
            