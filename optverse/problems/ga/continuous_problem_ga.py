from typing import Union, Dict, List, Tuple

import numpy as np

from ._problem_base_ga import ProblemBaseGA
from optverse._core.ga.operators.continuous.population import InitializePopulationContinuous
from optverse._core.ga.operators.continuous.crossovers import crossover_combination_extrapolation
from optverse._core.ga.operators.continuous.mutation import mutation_proba


class ContinuousProblemGA(ProblemBaseGA):
    
    def __init__(self, *, 
                 pop_size: int,
                 vector_size: int, 
                 min_values_vector: Union[int, float, np.ndarray] = None, 
                 max_values_vector: Union[int, float, np.ndarray] = None, 
                 loc_values_vector: Union[int, float, np.ndarray] = None, 
                 scale_values_vector: Union[int, float, np.ndarray] = None,
                 selection_k: int = None,
                 mutation_rate: float = 0.1,
                 mutation_incidence: float =0.2,
                 **kwargs):
        
        super().__init__(pop_size=pop_size, vector_size=vector_size, selection_k=selection_k,
                         mutation_rate=mutation_rate, mutation_incidence=mutation_incidence, **kwargs)
        
        self.min_values_vector = min_values_vector
        self.max_values_vector = max_values_vector
        self.loc_values_vector = loc_values_vector
        self.scale_values_vector = scale_values_vector
        

    def initialize_population(self, *args, **kwargs) -> List[Dict]:
        pop_initializer = InitializePopulationContinuous(**self.__dict__ , **kwargs)
        population = pop_initializer.generate_population(self.pop_size, *args, **kwargs)
        return population


    def _evaluate(self, x: np.ndarray, *args, **kwargs) -> float:
        return np.sum(x)


    def crossover(self, parents: Tuple[Dict], *args, **kwargs) -> Tuple[Dict]:
        return crossover_combination_extrapolation(parents, *args, **kwargs)


    def mutate(self, solution: Dict, *args, **kwargs) -> Dict:
        mutated_solution = mutation_proba(solution, self.mutation_rate, self.mutation_incidence, 
                                          self.min_values_vector, 
                                          self.max_values_vector, 
                                          self.scale_values_vector, *args, **kwargs)
        return mutated_solution

        