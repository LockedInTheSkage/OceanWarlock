import numpy as np
import pandas as pd

def categorize_navstat(toy_data):
    important_values = {0: "OMW_E", 1: "Anchor", 5: "Moored"}

    # Create a new column with categories
    return_value=toy_data['navstat'].apply(lambda x: important_values.get(x, "Other"))
    return_value.name = "navstat_cat"
    return return_value

def categorize_rot(total_df):
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

    #I want to turn the ISO into a numerical value.
    #I will do this by taking a list of all the unique values and then assign a number to each of them.
    #I will then create a dictionary with the values and the corresponding number.
    #I will then use the map function to apply this to the column.

    unique_values = total_df["ISO"].unique()
    unique_values_dict = {value: i for i, value in enumerate(unique_values)}
    
    return_values = total_df["ISO"].map(unique_values_dict)
    return_values.name = "ISO_num"
    return return_values

def days_to_etaParsed(total_df):
    eta=total_df["etaParsed"]
    time=total_df["time"]
    print(eta)
    print(time)
    difference = eta-time
    print(difference)
    return_values = (total_df["etaParsed"] - total_df["time"]).dt.days
    return_values.name = "days_to_etaParsed"
    return return_values

