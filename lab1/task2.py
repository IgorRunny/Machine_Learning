import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def estimate_slope(x_vals, y_vals):
    n = len(x_vals)
    sum_x = x_vals.sum()
    sum_y = y_vals.sum()
    sum_xy = (x_vals * y_vals).sum()
    sum_x2 = sum_x**2
    sum_x_sq = (x_vals**2).sum()

    return (sum_x * sum_y / n - sum_xy) / (sum_x2 / n - sum_x_sq)


def estimate_intercept(x_vals, y_vals, slope):
    mean_x = x_vals.mean()
    mean_y = y_vals.mean()
    return mean_y - slope * mean_x


def linear_prediction(x, intercept, slope):
    return intercept + slope * x


if __name__ == '__main__':
    data = datasets.load_diabetes()
    features = data.data
    targets = data.target

    print("Available features:", data.feature_names)

    idx_bp = data.feature_names.index('bp')
    bp_values = features[:, idx_bp].reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(bp_values,
                                                        targets,
                                                        test_size=0.2,
                                                        random_state=0)

    skl_model = LinearRegression()
    skl_model.fit(X_train, y_train)
    preds_skl = skl_model.predict(X_test)

    print(f"Scikit-Learn: slope = {skl_model.coef_[0]:.4f}, intercept = {skl_model.intercept_:.4f}")

    x_flat = X_train.flatten()
    y_flat = y_train
    slope_manual = estimate_slope(pd.Series(x_flat), pd.Series(y_flat))
    intercept_manual = estimate_intercept(pd.Series(x_flat), pd.Series(y_flat), slope_manual)

    preds_manual = [linear_prediction(xi[0], intercept_manual, slope_manual) for xi in X_test]

    print(f"Manual: slope = {slope_manual:.4f}, intercept = {intercept_manual:.4f}")

    plt.scatter(bp_values, targets, label='Data points')
    plt.plot(X_test, preds_skl, label='SKL prediction', linewidth=2)
    plt.plot(X_test, preds_manual, linestyle='--', label='Manual prediction')
    plt.xlabel('Mean Blood Pressure')
    plt.ylabel('Disease Progression')
    plt.title('Diabetes Regression Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    comparison_df = pd.DataFrame({
        'Actual': y_test,
        'SKLearn': preds_skl,
        'Manual': preds_manual
    })
    print(comparison_df)