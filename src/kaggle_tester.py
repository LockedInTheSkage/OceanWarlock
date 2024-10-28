
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from feature_engineering.tensor_features import develop_features, floating_conv
from path_finder import test_path_sorter, path_sorter

from sklearn.preprocessing import StandardScaler
from data_handler import LocalToLargeDataLoader


data_loader = LocalToLargeDataLoader(print_progress=True)
parsed_data = data_loader.load_raw_data(path="../../resources")
test_data = data_loader.load_test_data(path="../../resources")


new_data = test_path_sorter(parsed_data, test_data)
print(new_data.keys())


cols = new_data.columns.tolist()

time_0_index = cols.index("time_0")
time_1_index = cols.index("time_1")
print(time_0_index, time_1_index)

cols = cols[:time_1_index] + [cols[time_0_index]] + cols[time_1_index:-1]

X_new = new_data[cols]

print(X_new.keys())
print(X_new.shape)


test_data=develop_features(X_new)

model = load_model('B221024L2650.keras')


predictions = model.predict(X_new.to_numpy())  # Ensure the input is a numpy array

# Assuming predictions are in the form of (longitude, latitude) for a regression task
longitude_predicted = predictions[:, 0]  # First column corresponds to predicted longitude
latitude_predicted = predictions[:, 1]  # Second column corresponds to predicted latitude

# Step 3: Create a new DataFrame with the ID and predictions
predictions_df = pd.DataFrame({
    'ID': test_data['ID'],
    'longitude_predicted': longitude_predicted,
    'latitude_predicted': latitude_predicted
})

