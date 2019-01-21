# -*- coding: utf-8 -*-
from metaheuristics.optimization_solution import OptimizationSolution


def test_can_create_optimization_solution():
    solution = OptimizationSolution((2, 3), 15.5)
    assert solution.get_solution_value() == (2, 3)
    assert solution.get_objective_value() == 15.5
