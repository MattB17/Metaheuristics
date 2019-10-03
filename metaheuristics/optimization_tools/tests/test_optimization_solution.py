from metaheuristics.optimization_tools.optimization_solution \
    import OptimizationSolution


def test_can_create_optimization_solution_with_tuple():
    solution = OptimizationSolution((2, 3), 15.5)
    assert solution.get_solution_value() == (2, 3)
    assert solution.get_objective_value() == 15.5


def test_can_create_optimization_solution_with_list():
    solution = OptimizationSolution(['a', 'b', 'c'], 12.7)
    assert solution.get_solution_value() == ['a', 'b', 'c']
    assert solution.get_objective_value() == 12.7
