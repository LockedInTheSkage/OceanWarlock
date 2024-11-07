import numpy as np
import pandas as pd

def categorize_navstat(toy_data):
    if "navstat" not in toy_data.columns:
        return None
    movement_values = [0, 3, 4, 8]

    # Create a new column with categories
    toy_data["navstat_cat"] = np.where(toy_data["navstat"].isin(movement_values), "moving","not_moving")
    return_value = toy_data["navstat_cat"]
    return_value.name = "navstat_cat"
    return return_value

def categorize_rot(total_df):
    if "rot" not in total_df.columns:
        return None
    # Create a new column with categories
    rot_values = total_df["rot"]

    #This show me that I have ROT-values that are quite small, most are between 0 and -40 (by eyesight). I want to make this into a categorical based on what I see.
    #But be careful about the y-axis I have a big amount of values other than between 0 and -40.

    #Want to make this into a categorical.
    #I want my bins to be [-127, -126, 0, 126, np.inf]

    bins = [-127, -126, 0, 127, np.inf] #Based on the interpretation of the ROT and the histogram.

    labels = ["Port_T_S", "Port_T_L", "Star_T_L", "Star_T_S"]

    return_values = pd.cut(rot_values, bins = bins, labels = labels, right = False)
    return_values.name = "rot_cat"
    return return_values

def numerize_UN_LOCODE(total_df):
    if "UN_LOCODE" not in total_df.columns:
        return None

    #I want to turn the UN_LOCODE into a numerical value.
    #I will do this by taking a list of all the unique values and then assign a number to each of them.
    #I will then create a dictionary with the values and the corresponding number.
    #I will then use the map function to apply this to the column.

    unique_values = total_df["UN_LOCODE"].unique()
    unique_values_dict = {value: i for i, value in enumerate(unique_values)}
    
    return_values = total_df["UN_LOCODE"].map(unique_values_dict)
    return_values.name = "UN_LOCODE_num"
    return return_values

def numerize_ISO(total_df):
    if "ISO" not in total_df.columns:
        return None

    #I want to turn the ISO into a numerical value.
    #I will do this by taking a list of all the unique values and then assign a number to each of them.
    #I will then create a dictionary with the values and the corresponding number.
    #I will then use the map function to apply this to the column.

    unique_values = total_df["ISO"].unique()
    unique_values_dict = {value: i for i, value in enumerate(unique_values)}
    
    return_values = total_df["ISO"].map(unique_values_dict)
    return_values.name = "ISO_num"
    return return_values

def type_dummies(total_df):
    if "vesselType" not in total_df.columns:
        return None

    dummies = pd.get_dummies(total_df["vesselType"], dummy_na=True, prefix='value')
    return_values = pd.concat([total_df, dummies], axis = 1)
    return return_values

def days_to_etaParsed(total_df):
    if "etaParsed" not in total_df.columns or "time" not in total_df.columns:
        return None
    eta=total_df["etaParsed"]
    time=total_df["time"]
    print(eta)
    print(time)
    difference = eta-time
    print(difference)
    return_values = (total_df["etaParsed"] - total_df["time"]).dt.days
    return_values.name = "days_to_etaParsed"
    return return_values

