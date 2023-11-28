from typing import Dict, List, Tuple
import random


def parenting(solutions: List[Dict], n_pairs: int = None, *args, **kwargs) -> List[Tuple[Dict]]:
    
    solution_pairs = []
    
    if n_pairs == None:
        n_pairs = len(solutions)

    for _ in range(n_pairs):
        indices = random.sample(range(len(solutions)), 2)
        index1, index2 = indices[0], indices[1]
        solution_pairs += (solutions[index1], solutions[index2]),
    
    return solution_pairs


