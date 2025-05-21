import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn import datasets


def split_csv(path: str, target_idx: int, test_size: float = 0.2, random_state: int = 0):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, target_idx].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def split_diabetes(feature_name: str, test_size: float = 0.2, random_state: int = 0):
    data = datasets.load_diabetes()
    features = data.feature_names
    idx = features.index(feature_name)
    X = data.data[:, idx].reshape(-1, 1)
    y = data.target
    return train_test_split(X, y, test_size=test_size, random_state=random_state), features


def train_model(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


def display_evaluation(y_true, y_pred, label: str):
    results = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    print(f"\n{label} evaluation:")
    print(results)
    print(f"R2:   {r2_score(y_true, y_pred):.4f}")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred):.4f}\n")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_csv('J:\Machine_Learning\ML_labs\lab1\student_scores.csv', target_idx=1)
    model_stud, preds_stud = train_model(X_train, y_train, X_test)
    display_evaluation(y_test, preds_stud, 'Student Scores')

    (X_train_d, X_test_d, y_train_d, y_test_d), feature_names = split_diabetes('bp')
    print("Diabetes features:", feature_names)
    model_diab, preds_diab = train_model(X_train_d, y_train_d, X_test_d)
    display_evaluation(y_test_d, preds_diab, "Diabetes BP")
