from typing import Dict, List
import random


def tournament_selection(solutions: List[Dict], tournament_size: int = None, k: int = None) -> List[Dict]:

    if tournament_size is None:
        tournament_size = len(solutions)

    selected_solutions = []

    if k is None:
        k = max(3, round(tournament_size * 0.1))
    
    k = min(tournament_size, k)

    filtered_solutions = [s for s in solutions if s['is_feasible']]
    n_filtered_solutions = len(filtered_solutions)

    if n_filtered_solutions == 0:
        filtered_solutions = solutions
        n_filtered_solutions = len(solutions)

    if n_filtered_solutions < k:
        additional_solutions = random.choices(filtered_solutions, k=k - n_filtered_solutions)
        filtered_solutions.extend(additional_solutions)

    for _ in range(tournament_size):
        candidates = random.sample(filtered_solutions, k)
        best_solution_i = min(candidates, key=lambda x: x['optimization_eval'])
        selected_solutions += best_solution_i,

    return selected_solutions

