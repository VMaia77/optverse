from typing import Dict, List
from optverse.problems._problem_base import ProblemBase


class SearcherBase:

    def __init__(self, problem: ProblemBase, *args, **kwargs):
        self.problem = problem
        self.history_best = []
        self.history_worse = []
        self.best_solution, self.best_eval = None, float('inf')
        self.worst_solution, self.worst_eval = None, float('-inf')


    def evaluate(self, solutions: List[Dict], *args, **kwargs)  -> None:
        [self.problem.evaluate(s, *args, **kwargs) for s in solutions]


    def rank_solutions(self, solutions: List[Dict], *args, **kwargs) -> List[Dict]:
        ranked_solution_and_evals = sorted(solutions, key=lambda x: x['optimization_eval'])
        return ranked_solution_and_evals


    def get_best_worse_solutions_and_evals(self, solutions: List[Dict], *args, **kwargs) -> Dict:

        _solutions = [s for s in solutions if s['is_feasible']]

        if len(_solutions) == 0:
            _solutions = solutions.copy()

        ranked_solution_and_evals = self.rank_solutions(_solutions, *args, **kwargs)
        best_eval = ranked_solution_and_evals[0]['optimization_eval']
        worst_eval = ranked_solution_and_evals[-1]['optimization_eval']

        best_worst_solutions_and_evals = {}

        best_worst_solutions_and_evals['best_solution'] = ranked_solution_and_evals[0]
        best_worst_solutions_and_evals['best_eval'] = best_eval
        best_worst_solutions_and_evals['worst_solution'] = ranked_solution_and_evals[-1]
        best_worst_solutions_and_evals['worst_eval'] = worst_eval

        return best_worst_solutions_and_evals
    

    def update(self, solutions: List[Dict], save_history: bool, *args, **kwargs) -> None:
        
        best_worst_solutions_and_evals = self.get_best_worse_solutions_and_evals(solutions, *args, **kwargs)

        best_solution = best_worst_solutions_and_evals['best_solution']
        best_eval = best_worst_solutions_and_evals['best_eval']
        worst_solution = best_worst_solutions_and_evals['worst_solution']
        worst_eval = best_worst_solutions_and_evals['worst_eval']

        if best_eval < self.best_eval:
            self.best_solution = best_solution
            self.best_eval = best_eval
        if worst_eval > self.worst_eval:
            self.worst_solution = worst_solution
            self.worst_eval = worst_eval
    
        if save_history:
            self.history_best += best_worst_solutions_and_evals['best_eval'],
            self.history_worse += best_worst_solutions_and_evals['worst_eval'],
