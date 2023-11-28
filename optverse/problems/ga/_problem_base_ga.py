from typing import Dict, List, Tuple

from optverse.problems._problem_base import ProblemBase
from optverse._core.ga.operators.selection import tournament_selection
from optverse._core.ga.operators.parenting import parenting


class ProblemBaseGA(ProblemBase):

    def __init__(self, *, pop_size: int, vector_size: int, selection_k: int = None, 
                 mutation_rate: float = 0.1, mutation_incidence: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.vector_size = vector_size
        self.selection_k = selection_k
        self.mutation_rate = mutation_rate
        self.mutation_incidence = mutation_incidence

    def select(self, solutions: List[Dict], pop_size: int, *args, **kwargs) -> List[Dict]:
        return tournament_selection(solutions, pop_size, k=self.selection_k, *args, **kwargs)

    def parenting(self, solutions: List[Dict], *args, **kwargs)  -> List[Tuple[Dict]]:
        return parenting(solutions, n_pairs=None, *args, **kwargs)