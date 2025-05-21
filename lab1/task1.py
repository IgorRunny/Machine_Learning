import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


def compute_slope(x_vals, y_vals):
    n = len(x_vals)
    total_x = sum(x_vals)
    total_y = sum(y_vals)
    total_xy = sum(x_vals[i] * y_vals[i] for i in range(n))
    total_x_sq = total_x ** 2
    sum_x2 = sum(x ** 2 for x in x_vals)

    return (total_x * total_y / n - total_xy) / (total_x_sq / n - sum_x2)


def compute_intercept(x_vals, y_vals, slope):
    mean_x = sum(x_vals) / len(x_vals)
    mean_y = sum(y_vals) / len(y_vals)
    return mean_y - slope * mean_x


def predict(x, intercept, slope):
    return intercept + slope * x


def summarize_stats(name_x, name_y, x_vals, y_vals):
    count_x, count_y = len(x_vals), len(y_vals)
    stats = {
        'count': (count_x, count_y),
        'min': (min(x_vals), min(y_vals)),
        'max': (max(x_vals), max(y_vals)),
        'mean': (sum(x_vals)/count_x, sum(y_vals)/count_y)
    }
    print(f"Статистика столбцов {name_x} и {name_y}:")
    print(f"  количество: {name_x}={stats['count'][0]}, {name_y}={stats['count'][1]}")
    print(f"  min:        {name_x}={stats['min'][0]}, {name_y}={stats['min'][1]}")
    print(f"  max:        {name_x}={stats['max'][0]}, {name_y}={stats['max'][1]}")
    print(f"  среднее:    {name_x}={stats['mean'][0]}, {name_y}={stats['mean'][1]}\n")


if __name__ == '__main__':
    path = input("Укажите путь к CSV-файлу: ")
    data = pd.read_csv(Path(path))
    col_a, col_b = data.columns[:2]

    choice = int(input(f"Выберите, какая колонка по оси X и какая по оси Y:\n"
                       f"1) X: {col_a}, Y: {col_b}\n"
                       f"2) X: {col_b}, Y: {col_a}\n                       "))

    if choice == 1:
        x_data = data[col_a].tolist()
        y_data = data[col_b].tolist()
        x_label, y_label = col_a, col_b
    elif choice == 2:
        x_data = data[col_b].tolist()
        y_data = data[col_a].tolist()
        x_label, y_label = col_b, col_a
    else:
        raise ValueError("Неверный выбор осей.")

    summarize_stats(x_label, y_label, x_data, y_data)

    slope = compute_slope(x_data, y_data)
    intercept = compute_intercept(x_data, y_data, slope)
    fitted = [predict(x, intercept, slope) for x in x_data]

    fig = plt.figure(figsize=(12, 8))
    spec = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(spec[0, 0])
    ax1.scatter(x_data, y_data)
    ax1.set(title="Scatter Plot", xlabel=x_label, ylabel=y_label)

    ax2 = fig.add_subplot(spec[0, 1])
    ax2.scatter(x_data, y_data)
    ax2.plot(x_data, fitted, "r-")
    ax2.set(title="Regression Line", xlabel=x_label, ylabel=y_label)

    ax3 = fig.add_subplot(spec[1, 0])
    ax3.scatter(x_data, y_data)
    ax3.plot(x_data, fitted, "r-")
    for xi, yi, y_pred in zip(x_data, y_data, fitted):
        deviation = y_pred - yi
        rect = Rectangle((xi, yi), deviation, deviation,
                         linewidth=1, edgecolor='green', fill=False, hatch='///')
        ax3.add_patch(rect)
    ax3.set(title="Deviations", xlabel=x_label, ylabel=y_label)

    plt.tight_layout()
    plt.show()
