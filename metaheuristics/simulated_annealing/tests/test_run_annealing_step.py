# -*- coding: utf-8 -*-

from metaheuristics.simulated_annealing.simulated_annealing_utils \
    import run_annealing_step
from unittest.mock import patch, MagicMock


def test_when_solution_changes():
    mock_optimization_problem = MagicMock()
    mock_curr_solution = MagicMock()
    mock_new_solution = MagicMock()
    mock_optimization_problem.get_current_solution = MagicMock(return_value=mock_curr_solution)
    mock_optimization_problem.find_neighbour_solution = MagicMock(return_value=mock_new_solution)
    mock_curr_solution.get_objective_value = MagicMock(return_value=12)
    mock_new_solution.get_objective_value = MagicMock(return_value=9)
    with patch("metaheuristics.simulated_annealing.simulated_annealing_utils.should_change_solution",
               return_value=True) as mock_decision:
        result = run_annealing_step(1200, mock_optimization_problem)
    mock_optimization_problem.get_current_solution.assert_called_once()
    mock_optimization_problem.find_neighbour_solution.assert_called_once()
    mock_decision.assert_called_once_with(12, 9, 1200)
    assert result == mock_new_solution


def test_when_solution_does_not_change():
    mock_optimization_problem = MagicMock()
    mock_curr_solution = MagicMock()
    mock_new_solution = MagicMock()
    mock_optimization_problem.get_current_solution = MagicMock(return_value=mock_curr_solution)
    mock_optimization_problem.find_neighbour_solution = MagicMock(return_value=mock_new_solution)
    mock_curr_solution.get_objective_value = MagicMock(return_value=17.5)
    mock_new_solution.get_objective_value = MagicMock(return_value=37)
    with patch("metaheuristics.simulated_annealing.simulated_annealing_utils.should_change_solution",
               return_value=False) as mock_decision:
        result = run_annealing_step(150, mock_optimization_problem)
    mock_optimization_problem.get_current_solution.assert_called_once()
    mock_optimization_problem.find_neighbour_solution.assert_called_once()
    mock_decision.assert_called_once_with(17.5, 37, 150)
    assert result == mock_curr_solution
