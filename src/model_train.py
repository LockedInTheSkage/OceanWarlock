import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ModelTrain:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestClassifier()
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        self.data.dropna(inplace=True)

    def preprocess_data(self):
        # Assuming the last column is the target variable
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred)
        return accuracy, precision, recall, f1, roc_auc

    def plot_feature_importances(self):
        feature_importances = self.model.feature_importances_
        sns.barplot(x=feature_importances, y=self.data.columns[:-1])
        plt.title('Feature Importances')
        plt.show()