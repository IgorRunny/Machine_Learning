import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


def visualize_iris(data, features, targets, names):
    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(12, 5))

    # Sepal dimensions
    plt.subplot(1, 2, 1)
    for idx, color in enumerate(colors):
        plt.scatter(
            features[idx*50:(idx+1)*50, 0],
            features[idx*50:(idx+1)*50, 1],
            color=color, label=names[idx]
        )
    plt.xlabel('Sepal length'); plt.ylabel('Sepal width')
    plt.title('Sepal Length vs Width')
    plt.legend()

    # Petal dimensions
    plt.subplot(1, 2, 2)
    for idx, color in enumerate(colors):
        plt.scatter(
            features[idx*50:(idx+1)*50, 2],
            features[idx*50:(idx+1)*50, 3],
            color=color, label=names[idx]
        )
    plt.xlabel('Petal length'); plt.ylabel('Petal width')
    plt.title('Petal Length vs Width')
    plt.legend()
    plt.tight_layout()
    plt.show()


def pairplot_iris(features, targets, feature_names):
    df = pd.DataFrame(features, columns=feature_names)
    df['species'] = [datasets.load_iris().target_names[i] for i in targets]
    sns.pairplot(df, hue='species')
    plt.show()


def train_logistic(X, y, label=None):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LogisticRegression(random_state=0)
    model.fit(X_tr, y_tr)
    predictions = model.predict(X_te)
    accuracy = model.score(X_te, y_te)

    if label:
        print(f"\n--- {label} ---")
    print(f"Predicted: {predictions}")
    print(f"Actual:    {y_te}")
    print(f"Accuracy:  {accuracy:.4f}\n")

    return model, (X_tr, X_te, y_tr, y_te)


def plot_random_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.title('Synthetic Classification Data')
    plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
    plt.show()


if __name__ == '__main__':
    # Task 1 & 2: Iris visualization
    iris = datasets.load_iris()
    visualize_iris(iris, iris.data, iris.target, iris.target_names)
    pairplot_iris(iris.data, iris.target, iris.feature_names)

    # Task 3-8: Binary classification subsets
    subset_sv = (iris.data[iris.target != 2], iris.target[iris.target != 2])  # Setosa vs Versicolor
    subset_vv = (iris.data[iris.target != 0], iris.target[iris.target != 0])  # Versicolor vs Virginica

    train_logistic(*subset_sv, label='Setosa vs Versicolor')
    train_logistic(*subset_vv, label='Versicolor vs Virginica')

    # Task 9: Synthetic dataset
    X_syn, y_syn = make_classification(
        n_samples=1000, n_features=2, n_redundant=0,
        n_informative=2, random_state=1, n_clusters_per_class=1
    )
    plot_random_data(X_syn, y_syn)
    train_logistic(X_syn, y_syn, label='Synthetic Data')
