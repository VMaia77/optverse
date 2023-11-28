from typing import Callable
import numpy as np


def norm_decorator(norm_func: Callable):
    def wrapper(*vectors):
        if len(vectors) == 1:
            return norm_func(vectors[0])
        elif len(vectors) == 2:
            return norm_func(vectors[0] - vectors[1])
        else:
            raise ValueError("Invalid number of vectors provided. The decorator expects either one or two vectors.")
    return wrapper


@norm_decorator
def l1_norm(vector: np.ndarray):
    return np.linalg.norm(vector, ord=1)


@norm_decorator
def l2_norm(vector: np.ndarray):
    return np.linalg.norm(vector)


