# -*- coding: utf-8 -*-

from metaheuristics.optimization_problem import OptimizationProblem
from metaheuristics.optimization_solution import OptimizationSolution
from unittest.mock import call, MagicMock


def test_can_create():
    initial_solution = (1, 2)
    mock_objective_function = MagicMock(return_value=3)
    mock_updater = MagicMock()
    optimization_problem = OptimizationProblem(
            initial_solution, mock_objective_function, mock_updater)
    assert optimization_problem.get_objective_function() == mock_objective_function
    assert optimization_problem.get_current_solution() == initial_solution
    assert optimization_problem.get_current_objective_value() == 3
    mock_objective_function.assert_called_once_with(initial_solution)
    mock_updater.assert_not_called()


def test_find_neighbour_solution_once():
    initial_solution = (0, 1, 0)
    neighbour_solution = (-1, 1, 1)
    mock_objective_function = MagicMock(side_effect=[-10, 0.5])
    mock_updater = MagicMock(return_value=neighbour_solution)
    optimization_problem = OptimizationProblem(
            initial_solution, mock_objective_function, mock_updater)
    neighbour = optimization_problem.find_neighbour_solution()
    assert neighbour.get_solution_value() == neighbour_solution
    assert neighbour.get_objective_value() == 0.5
    assert mock_objective_function.call_count == 2
    objective_function_calls = [call(initial_solution), call(neighbour_solution)]
    mock_objective_function.assert_has_calls(objective_function_calls)
    mock_updater.assert_called_once_with(initial_solution)
    

def test_find_neighbour_solution_multiple_times():
    solution_vals = [["x", "y", "z"], ["y", "z", "x"], ["x", "z", "y"]]
    objective_vals = [2.3, 4.1, 0.7]
    mock_objective_function = MagicMock(side_effect=objective_vals)
    mock_updater = MagicMock(side_effect=solution_vals[1:])
    optimization_problem = OptimizationProblem(
            solution_vals[0], mock_objective_function, mock_updater)
    neighbour_1 = optimization_problem.find_neighbour_solution()
    assert neighbour_1.get_solution_value() == solution_vals[1]
    assert neighbour_1.get_objective_value() == objective_vals[1]
    neighbour_2 = optimization_problem.find_neighbour_solution()
    assert neighbour_2.get_solution_value() == solution_vals[2]
    assert neighbour_2.get_objective_value() == objective_vals[2]
    assert mock_objective_function.call_count == 3
    objective_func_calls = [call(solution) for solution in solution_vals]
    mock_objective_function.assert_has_calls(objective_func_calls)
    assert mock_updater.call_count == 2
    updater_calls = [call(solution_vals[0]), call(solution_vals[0])]
    mock_updater.assert_has_calls(updater_calls)
    
    
def test_update_current_solution_once():
    initial_solution_val = ["a", "b","c"]
    new_solution_val = ["b", "c", "a"]
    mock_objective_function = MagicMock(side_effect=[3.2, -12.9])
    mock_updater = MagicMock()
    optimization_problem = OptimizationProblem(
            initial_solution_val, mock_objective_function, mock_updater)
    assert optimization_problem.get_current_solution() == initial_solution_val
    assert optimization_problem.get_current_objective_value() == 3.2
    optimization_problem.update_current_solution(new_solution_val)
    assert optimization_problem.get_current_solution() == new_solution_val
    assert optimization_problem.get_current_objective_value() == -12.9
    
    
def test_update_current_solution_multiple_times():
    solution_vals = [(0, 1), (1, 1), (2, 1)]
    objective_vals = [2.3, 2.7, 1.9]
    mock_objective_function = MagicMock(side_effect=objective_vals)
    mock_updater = MagicMock()
    optimization_problem = OptimizationProblem(
            solution_vals[0], mock_objective_function, mock_updater)
    assert optimization_problem.get_current_solution() == solution_vals[0]
    assert optimization_problem.get_current_objective_value() == objective_vals[0]
    optimization_problem.update_current_solution(solution_vals[1])
    assert optimization_problem.get_current_solution() == solution_vals[1]
    assert optimization_problem.get_current_objective_value() == objective_vals[1]
    optimization_problem.update_current_solution(solution_vals[2])
    assert optimization_problem.get_current_solution() == solution_vals[2]
    assert optimization_problem.get_current_objective_value() == objective_vals[2]
    assert mock_objective_function.call_count == 3
    objective_func_calls = [call(solution) for solution in solution_vals]
    mock_objective_function.assert_has_calls(objective_func_calls)
    
