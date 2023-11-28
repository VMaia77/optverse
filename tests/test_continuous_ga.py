
import numpy as np
from optverse.searchers.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from optverse.problems.ga.continuous_problem_ga import ContinuousProblemGA
from optverse.optimization.optimizer import Optimizer


def test_cga_float_min_max():

    problem = ContinuousProblemGA(pop_size=50, 
                                  min_values_vector = 1,
                                  scale_values_vector = 3,
                                  vector_size=3, 
                                  sense='min')
    
    optimizer = Optimizer(problem, GeneticAlgorithm)

    solution = optimizer.optimize(100, save_history=False)
    assert solution['eval'] < -190, 'Problem in test_cga_float_min_max'


def test_cga_float_vector_min_max():

    min_vals = [1, 3, 7]
    max_vals = [10, 20, 30]

    vector_size = 3

    problem = ContinuousProblemGA(pop_size=50, 
                                  min_values_vector = np.array(min_vals),
                                  max_values_vector = np.array(max_vals),
                                  vector_size=vector_size, 
                                  sense='min')
    
    optimizer = Optimizer(problem, GeneticAlgorithm)

    solution = optimizer.optimize(100, save_history=False)
    for i in range(vector_size):
        assert solution['x'][i] > min_vals[i], 'Problem in test_cga_float_vector_min_max in min values'
        assert solution['x'][i] < max_vals[i], 'Problem in test_cga_float_vector_min_max in max values'


def test_cga_float_loc_scale():

    problem = ContinuousProblemGA(pop_size=50, 
                                  loc_values_vector = 7,
                                  scale_values_vector = 1,
                                  vector_size=3, 
                                  sense='min')
    
    optimizer = Optimizer(problem, GeneticAlgorithm)

    solution = optimizer.optimize(100, save_history=False)
    assert solution['eval'] < -45, 'Problem in test_cga_float_loc_scale'


def test_cga_float_vector_loc_scale():

    problem = ContinuousProblemGA(pop_size=50, 
                                  loc_values_vector = np.array([5, 50, 500]),
                                  scale_values_vector = np.array([0.1, 1, 5]),
                                  vector_size=3, 
                                  sense='min')
    
    optimizer = Optimizer(problem, GeneticAlgorithm)

    solution = optimizer.optimize(100, save_history=False)
    assert solution['eval'] < 400, 'Problem in test_cga_float_vector_loc_scale'


def test_cga_maximization():
    vector_size = 3
    min_vals = [5, 50, 500]
    max_val = 501
    problem = ContinuousProblemGA(pop_size=50, 
                                  min_values_vector = np.array(min_vals),
                                  max_values_vector = max_val,
                                  vector_size=vector_size, 
                                  sense='max')
    
    optimizer = Optimizer(problem, GeneticAlgorithm)

    solution = optimizer.optimize(100, save_history=False)
    for i in range(vector_size):
        assert solution['x'][i] > min_vals[i], 'Problem in test_cga_maximization in min values'
        assert solution['x'][i] < max_val, 'Problem in test_cga_maximization in max values'


def test_cga_constraint():

    problem = ContinuousProblemGA(pop_size=50, 
                                  vector_size=3, 
                                  sense='min')
    
    cval = -1000

    def c1(solution):
        if sum(solution['x']) < cval:
            return 1
        return 0

    problem.set_constraints([c1])

    optimizer = Optimizer(problem, GeneticAlgorithm)

    solution = optimizer.optimize(100, save_history=False)
    assert solution['eval'] > cval, 'Problem in test_cga_constraint'
    assert solution['is_feasible'], 'Problem in test_cga_constraint'

