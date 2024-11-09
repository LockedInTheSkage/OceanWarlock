import numpy as np
import pandas as pd

def categorize_navstat(toy_data):
    if "navstat" not in toy_data.columns:
        return None
    important_values = {0: "OMW_E", 1: "Anchor", 5: "Moored"}

    # Create a new column with categories
    return_values = toy_data['navstat'].apply(lambda x: important_values.get(x, "Other"))
    return_values = pd.get_dummies(return_values, dummy_na=True, prefix='value')
    return_values.columns = [f"navstat_{col}" for col in return_values.columns]
    toy_data = toy_data.drop(columns = "navstat")
    toy_data = pd.concat([toy_data, return_values], axis = 1)
    return toy_data

def categorize_rot(total_df):
    if "rot" not in total_df.columns:
        return None
 
    rot_values = total_df["rot"]

    bins = [-127, -126, 0, 127, np.inf]

    labels = ["Port_T_S", "Port_T_L", "Star_T_L", "Star_T_S"]

    return_values = pd.cut(rot_values, bins = bins, labels = labels, right = False)
    return_values = pd.get_dummies(return_values, dummy_na=True, prefix='value')
    return_values.columns = [f"rot_{col}" for col in return_values.columns]
    total_df = total_df.drop(columns = "rot")
    total_df = pd.concat([total_df, return_values], axis = 1)


    return total_df

def numerize_UN_LOCODE(total_df):
    if "UN_LOCODE" not in total_df.columns:
        return None

    unique_values = total_df["UN_LOCODE"].unique()
    unique_values_dict = {value: i for i, value in enumerate(unique_values)}
    
    return_values = total_df["UN_LOCODE"].map(unique_values_dict)
    total_df = total_df.drop(columns = "UN_LOCODE")
    total_df["UN_LOCODE_num"] = return_values

    return total_df

def numerize_vesselId(total_df):
    if "vesselId" not in total_df.columns:
        return None
    
    unique_values = total_df["vesselId"].unique()
    unique_values_dict = {value: i for i, value in enumerate(unique_values)}

    return_values = total_df["vesselId"].map(unique_values_dict)
    total_df = total_df.drop(columns = "vesselId")
    total_df["vesselId_num"] = return_values

    return total_df

def numerize_portId(total_df):
    if "portId" not in total_df.columns:
        return None

    unique_values = total_df["portId"].unique()
    unique_values_dict = {value: i for i, value in enumerate(unique_values)}
    
    return_values = total_df["portId"].map(unique_values_dict)
    total_df = total_df.drop(columns = "portId")
    total_df["portId_num"] = return_values

    return total_df

def numerize_shippingLineId(total_df):
    if "shippingLineId" not in total_df.columns:
        return None

    unique_values = total_df["shippingLineId"].unique()
    unique_values_dict = {value: i for i, value in enumerate(unique_values)}
    
    return_values = total_df["shippingLineId"].map(unique_values_dict)
    total_df = total_df.drop(columns = "shippingLineId")
    total_df["shippingLineId_num"] = return_values

    return total_df
    

def numerize_ISO(total_df):
    if "ISO" not in total_df.columns:
        return None

    unique_values = total_df["ISO"].unique()
    unique_values_dict = {value: i for i, value in enumerate(unique_values)}
    
    return_values = total_df["ISO"].map(unique_values_dict)
    
    total_df = total_df.drop(columns = "ISO")
    total_df["ISO_num"] = return_values
    return total_df

def numerize_homePort(total_df):
    if "homePort" not in total_df.columns:
        return None

    unique_values = total_df["homePort"].unique()
    unique_values_dict = {value: i for i, value in enumerate(unique_values)}
    
    return_values = total_df["homePort"].map(unique_values_dict)
    total_df = total_df.drop(columns = "homePort")
    total_df["homePort_num"] = return_values

    return total_df

def type_dummies(total_df):
    if "vesselType" not in total_df.columns:
        return None
    return_values = pd.get_dummies(total_df["vesselType"], dummy_na=True, prefix='value')
    return_values.columns = [f"vesselType_{col}" for col in return_values.columns]
    total_df = total_df.drop(columns = "vesselType")
    total_df = pd.concat([total_df, return_values], axis = 1)
    return total_df

def minutes_to_etaParsed(total_df):
    if "etaParsed" not in total_df.columns or "time" not in total_df.columns:
        return None
    return_values = (total_df["etaParsed"] - total_df["time"]).dt.days
    total_df = total_df.drop(columns = "etaParsed")
    total_df["days_to_etaParsed"] = return_values
    return total_df

