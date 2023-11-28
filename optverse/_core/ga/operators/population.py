from typing import Dict, List
import numpy as np


class InitializePopulation:

    def __init__(self, *args, **kwargs):
        ...

    def create_initial_individual(self, solution: np.ndarray) -> Dict:
        return {'x': solution}
    
    def create_initial_population(self, solutions: np.ndarray) -> List[Dict]:
        return [self.create_initial_individual(s) for s in solutions]