from typing import Union, Dict, List
import numpy as np
from ...operators.population import InitializePopulation


class InitializePopulationContinuous(InitializePopulation):

    def __init__(self, *, 
                 vector_size: int = None, 
                 min_values_vector: Union[int, float, np.ndarray] = None, 
                 max_values_vector: Union[int, float, np.ndarray] = None, 
                 loc_values_vector: Union[int, float, np.ndarray] = None, 
                 scale_values_vector: Union[int, float, np.ndarray] = None,
                 **kwargs):
        
        self.vector_size = vector_size

        self.min_values_vector = np.array(min_values_vector) if min_values_vector is not None else None
        self.max_values_vector = np.array(max_values_vector) if max_values_vector is not None else None
        self.loc_values_vector = np.array(loc_values_vector) if loc_values_vector is not None else None
        self.scale_values_vector = np.array(scale_values_vector) if scale_values_vector is not None else None


    def generate_population(self, pop_size: int) -> List[Dict]:

        if self.min_values_vector is not None and self.max_values_vector is not None:
            vectors = np.random.uniform(self.min_values_vector, self.max_values_vector, (pop_size, self.vector_size))
            return self.create_initial_population(vectors)
        
        if self.loc_values_vector is not None and self.scale_values_vector is not None:
            vectors = np.random.normal(self.loc_values_vector, self.scale_values_vector, (pop_size, self.vector_size))
            return self.create_initial_population(vectors)
                
        vectors = np.random.normal(0, 1, (pop_size, self.vector_size))

        return self.create_initial_population(vectors)