from typing import Dict, Tuple
import random
import numpy as np


def crossover_combination_extrapolation(parents: Tuple[Dict]) -> Tuple[Dict]:

    parent1, parent2 = parents[0], parents[1]
    vector_size = parent1['x'].shape[0]

    crossover_n_locs = random.randint(1, vector_size)
    crossover_locs_indexes = random.sample(range(vector_size), crossover_n_locs)

    offspring1_solution = np.copy(parent1['x'])
    offspring2_solution = np.copy(parent2['x'])

    for crossover_idx in crossover_locs_indexes:

        beta = np.random.uniform(0, 1, 1)

        new_value1 = parent1['x'][crossover_idx] - beta * (parent1['x'][crossover_idx] - parent2['x'][crossover_idx])
        new_value2 = parent2['x'][crossover_idx] + beta * (parent1['x'][crossover_idx] - parent2['x'][crossover_idx])

        offspring1_solution[crossover_idx] = new_value1
        offspring2_solution[crossover_idx] = new_value2

    offspring1, offspring2 = {'x': offspring1_solution}, {'x': offspring2_solution}
    
    return offspring1, offspring2