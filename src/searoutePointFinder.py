import searoute as sr
from datetime import timedelta
import pandas as pd
from shapely.geometry import LineString, Point

def fill_with_proximity(df):
    # Iterate over each segment of missing lat/lon data
    for start_idx, end_idx in find_missing_segments(df):
        # Get the last known and first known points before and after the gap
        start_point = df.loc[start_idx - timedelta(minutes=20), ['longitude', 'latitude']]
        end_point = df.loc[end_idx, ['longitude', 'latitude']]
        
        # Ensure both points are available
        if start_point.isna().any() or end_point.isna().any():
            continue

        # Calculate the route using searoute
        route = sr.searoute(start_point.tolist(), end_point.tolist())
        route['geometry']['coordinates'] = [start_point.tolist()] + route['geometry']['coordinates'] + [end_point.tolist()]
        # Interpolate route points for each missing timestamp in the segment
        fill_missing_segment(df, route, start_idx, end_idx)
    
    df = df.ffill()
    return df

def find_missing_segments(df):
    """
    Finds segments of missing latitude/longitude data in the DataFrame.
    Returns a list of tuples [(start_idx, end_idx), ...] for each missing segment.
    """
    missing_segments = []
    in_missing = False
    start_idx = None

    for idx, row in df.iterrows():
        if pd.isna(row['latitude']) and pd.isna(row['longitude']):
            if not in_missing:
                start_idx = idx
                in_missing = True
        else:
            if in_missing:
                end_idx = idx
                missing_segments.append((start_idx, end_idx))
                in_missing = False

    return missing_segments

def fill_missing_segment(df, route, start_idx, end_idx):
    """
    Fills in latitude and longitude data for a missing segment
    using interpolated points along the given route.
    """
    # Convert the route to a Shapely LineString for interpolation
    line = LineString(route['geometry']['coordinates'])
    num_missing = len(df[start_idx:end_idx])
    
    # Generate coordinates for each missing timestamp
    for i, idx in enumerate(df[start_idx:end_idx].index):
        fraction = (i + 1) / (num_missing + 1)
        point = line.interpolate(fraction * line.length)
        df.at[idx, 'latitude'] = point.y
        df.at[idx, 'longitude'] = point.x
