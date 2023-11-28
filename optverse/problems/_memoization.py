from typing import Dict, Callable


def memoize(evaluate_func: Callable) -> Callable:
    def wrapper(class_obj, solution: Dict, *args, **kwargs):
        solution_key = tuple(solution['x'])
        if solution_key in class_obj.memoization_cache:
            result = class_obj.memoization_cache[solution_key]
        else:
            result = evaluate_func(solution, *args, **kwargs)
            class_obj.memoization_cache[solution_key] = result
        return result
    return wrapper