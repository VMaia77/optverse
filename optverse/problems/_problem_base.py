
from typing import Union, Dict, List, Callable
from copy import copy


class ProblemBase:

    def __init__(self, *, sense: str = 'min', infeasiblity_penalty: Union[float, int] = 1e6, **kwargs):
        self.sense = sense
        self.memoization_cache = {}
        self.constraints = []
        self.infeasiblity_penalty = infeasiblity_penalty

    def set_constraints(self, constraints: List[Callable]) -> None:
        self.constraints = constraints

    def get_constraints(self) -> List[Callable]:
        return self.constraints
    
    def infeasible_handler(self, solution: Dict, cf: Callable) -> Dict:
        if cf(solution):
            solution['optimization_eval'] += self.infeasiblity_penalty # add penalty because it's always a minimization.
            solution['is_feasible'] = False
        return solution
    
    def evaluate(self, solution: Dict, *args, **kwargs) -> Dict:
        solution['optimization_eval'] = self._evaluate(solution['x'], *args, **kwargs)
        solution['eval'] = copy(solution['optimization_eval'])
        if self.sense != 'min':
            solution['optimization_eval'] *= -1
        solution['is_feasible'] = True
        for cf in self.get_constraints():
            solution = self.infeasible_handler(solution, cf)
        return solution
    
    def return_best_solution(self, best_solution) -> Dict:            
        best_solution['infeasible_in'] = []
        for cf in self.get_constraints():
            if cf(best_solution):
                best_solution['infeasible_in'] += cf.__name__,
        return best_solution
