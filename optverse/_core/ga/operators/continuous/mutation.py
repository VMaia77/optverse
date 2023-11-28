from typing import Union, Dict
import numpy as np
import random


def mutation_proba(solution: Dict, mutation_rate: float, 
                   mutation_incidence: float, 
                   min_values_vector: Union[int, float, np.ndarray] = None, 
                   max_values_vector: Union[int, float, np.ndarray] = None, 
                   scale_values_vector: Union[int, float, np.ndarray] = None, 
                   *args, **kwargs) -> Dict:

    if random.random() > mutation_rate:
        return solution
    
    vector_size = len(solution['x'])

    num_elements_to_mutate = max(1, int(mutation_incidence * vector_size))
    
    mutation_indices = np.random.choice(vector_size, size=num_elements_to_mutate, replace=False)
    
    mutation_indices.sort()

    mutated_solution = solution.copy()

    for index in mutation_indices:

        if min_values_vector is not None and max_values_vector is not None:
            min_val = min_values_vector[index] if isinstance(min_values_vector, np.ndarray) else min_values_vector
            max_val = max_values_vector[index] if isinstance(max_values_vector, np.ndarray) else max_values_vector
            mutated_solution['x'][index] = np.random.uniform(min_val, max_val)
            continue

        if scale_values_vector is not None:
            scale_val = scale_values_vector[index] if isinstance(scale_values_vector, np.ndarray) else scale_values_vector
            mutated_solution['x'][index] = np.random.normal(mutated_solution['x'][index], scale_val)
            continue

        mutated_solution['x'][index] = np.random.normal(mutated_solution['x'][index], 
                                                        np.abs(mutated_solution['x'][index]))

    return mutated_solution
