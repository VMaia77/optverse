from typing import Dict, List

from itertools import chain
from tqdm import tqdm

from optverse.searchers._searcher_base import SearcherBase


class GeneticAlgorithm(SearcherBase):

    def select(self, solutions: List[Dict], pop_size: int, *args, **kwargs) -> List[Dict]:
        return self.problem.select(solutions, pop_size, *args, **kwargs)

    def crossover(self, solutions: List[Dict], *args, **kwargs) -> List[Dict]:
        parents_grouped = self.problem.parenting(solutions, *args, **kwargs)
        generated_population = list(map(self.problem.crossover, parents_grouped))
        generated_population = list(chain.from_iterable(generated_population))
        return generated_population
    
    def mutate(self, solutions: List[Dict], *args, **kwargs) -> List[Dict]:
        mutated_solutions = list(map(lambda solution: self.problem.mutate(solution, *args, **kwargs), solutions))
        return mutated_solutions

    def _search(self, n_gen: int, save_history: bool = True, *args, **kwargs) -> Dict:

        population = self.problem.initialize_population()
        self.evaluate(population, *args, **kwargs)

        for gen in tqdm(range(n_gen)):
            self.update(population, save_history, *args, **kwargs)
            selected_population = self.select(population, self.problem.pop_size, *args, **kwargs)
            generated_population = self.crossover(selected_population, *args, **kwargs)
            mutated_population = self.mutate(generated_population, *args, **kwargs)
            population = selected_population + mutated_population
            self.evaluate(population, *args, **kwargs)

        self.update(population, save_history, *args, **kwargs)

        return self.problem.return_best_solution(self.best_solution)