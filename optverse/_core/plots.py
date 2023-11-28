from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt



def plot_history(history_best: List[Union[int, float]], history_worse: List[Union[int, float]] = []) -> None:

    iterations = list(range(1, len(history_best) + 1))

    plt.plot(iterations, history_best, label='Best evals', marker='o')
    
    if history_worse:
        plt.plot(iterations, history_worse, label='Worst evals', marker='x')

    plt.xlabel('Iteration')
    plt.ylabel('Eval')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_2d_space(evaluate_function, search_range_x1_min, search_range_x1_max, search_range_x2_min, search_range_x2_max, points=[]):
    x = np.linspace(search_range_x1_min, search_range_x1_max, 100)
    y = np.linspace(search_range_x2_min, search_range_x2_max, 100)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack((xx.ravel(), yy.ravel())).T
    values = np.array([evaluate_function(point) for point in grid])
    values = values.reshape(xx.shape)

    plt.imshow(values, cmap='viridis', extent=[search_range_x1_min, search_range_x1_max, search_range_x2_min, search_range_x2_max], origin='lower')
    
    for point in points:
        plt.plot(point[0], point[1], marker='o', markersize=7, markeredgewidth=2, color='red')

    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Space Plot')
    plt.show()