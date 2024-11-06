import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from feature_engineering.tensor_features import develop_features, floating_conv
from path_finder import path_sorter

from sklearn.preprocessing import StandardScaler
from data_handler import LocalToLargeDataLoader



data_loader = LocalToLargeDataLoader(print_progress=True)
parsed_data = data_loader.load_raw_data(path="../../resources")


print(parsed_data.keys())


path_dict = path_sorter(parsed_data)


print(list(path_dict.keys()))
print(path_dict["time_key"][0])
print(path_dict['navstat'])
print(path_dict[list(path_dict.keys())[0]][0])

features, y, test_features, y_test = data_loader.load_training_data(path_dict) 


features=develop_features(features)
test_features=develop_features(test_features)


print(list(features.keys()))
print(len(list(features.keys())))
# print(features.head())
print(y.head())


X_train = features.to_numpy()
X_test_np = test_features.to_numpy()
y_train = y.to_numpy()
y_test_np = y_test.to_numpy()


model = Sequential()

# Input layer
input_dim = X_train.shape[1]  # Number of features in your dataset

# Add layers
model.add(Dense(64, input_dim=input_dim, activation='relu'))  # First hidden layer
model.add(Dropout(0.3))  # Dropout for regularization
model.add(Dense(32, activation='relu'))  # Second hidden layer
model.add(Dense(16, activation='relu'))  # Third hidden layer

# Output layer
# If it's a regression task (predicting a continuous variable like time):
model.add(Dense(units=2, activation='linear')) 

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])



# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)


history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                    validation_split=0.2, callbacks=[early_stopping, model_checkpoint])


loss, mae = model.evaluate(X_test_np, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")