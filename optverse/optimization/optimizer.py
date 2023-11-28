from typing import Type, TypeVar, Dict
import time
from optverse._core.plots import plot_history
from optverse.problems._problem_base import ProblemBase
from optverse.searchers._searcher_base import SearcherBase


T = TypeVar('T', bound=SearcherBase)


class Optimizer:

    def __init__(self, problem: ProblemBase, searcher: Type[T], *args, **kwargs):
        self.problem = problem
        self.searcher = searcher(self.problem, *args, **kwargs)

    def optimize(self, n_iter: int, save_history: bool = True, *args, **kwargs) -> Dict:
        t0 = time.time()
        solution = self.searcher._search(n_iter, save_history, *args, **kwargs)
        runtime = time.time() - t0
        solution['runtime'] = runtime
        return solution
    
    def plot_history(self, include_worst: bool = False):
        if include_worst:
            plot_history(self.searcher.history_best, self.searcher.history_worse)
        else:
            plot_history(self.searcher.history_best)