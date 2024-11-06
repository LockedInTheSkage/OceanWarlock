import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pandas as pd

from global_land_mask import globe

def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return np.floor(n * multiplier) / multiplier


def find_land_points(lat, lon, n=8, step_size=0.01, patience=20, max_distance=5):
    
    closest_land_points = []

    angle = 360 / n

    for i in range(n):

        hits = 0
        check_lat = lat
        check_lon = lon
        hit_location = None

        while hits < patience:
            if globe.is_land(check_lat, check_lon):
                if hit_location is None:
                    hit_location = (check_lat, check_lon)
                hits += 1
            else:
                hit_location = None
                hits = 0
            check_lat += step_size*np.sin(np.radians(angle*i))
            check_lon += step_size*np.cos(np.radians(angle*i))
            if np.sqrt((check_lat-lat)**2 + (check_lon-lon)**2) > max_distance:
                hit_location = (check_lat, check_lon)
                break
        closest_land_points.append((angle*i, np.sqrt((hit_location[0]-lat)**2 + (hit_location[1]-lon)**2), hit_location))
    

    return closest_land_points

def find_land_points_df(df, lat_col="latitude_t", lon_col="longitude_t", n=8, step_size=0.01, patience=20, max_distance=5):

    def apply_find_land_points(row):
        lat = row[lat_col]
        lon = row[lon_col]
        return find_land_points(lat, lon, n, step_size, patience, max_distance)
    
    closest_land_points = df.apply(apply_find_land_points, axis=1)
    
    # Create a DataFrame with the closest_land_points
    columns = [f'point_{i}' for i in range(n)]
    closest_land_points_df = pd.DataFrame(closest_land_points.tolist(), columns=columns)
    
    return closest_land_points_df