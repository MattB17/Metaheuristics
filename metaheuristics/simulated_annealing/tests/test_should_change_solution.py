# -*- coding: utf-8 -*-

from ..simulated_annealing_utils import should_change_solution
from unittest.mock import patch, MagicMock


@patch("metaheuristics.simulated_annealing.simulated_annealing_utils.simulated_annealing_function")
def test_with_better_objective_value(mock_annealing_func, monkeypatch):
    mock_random = MagicMock()
    monkeypatch.setattr("numpy.random.rand", mock_random)
    assert should_change_solution(25, 20, 300)
    mock_annealing_func.assert_not_called()
    mock_random.assert_not_called()
    
    
@patch("metaheuristics.simulated_annealing.simulated_annealing_utils.simulated_annealing_function",
       return_value=0.75)
def test_when_random_num_less_than_func_val(mock_annealing_func, monkeypatch):
    mock_random = MagicMock(return_value=0.5)
    monkeypatch.setattr("numpy.random.rand", mock_random)
    assert should_change_solution(10, 12.5, 300)
    mock_annealing_func.assert_called_once_with(10, 12.5, 300)
    mock_random.assert_called_once()


@patch("metaheuristics.simulated_annealing.simulated_annealing_utils.simulated_annealing_function",
       return_value=0.67)
def test_when_random_num_equals_func_val(mock_annealing_func, monkeypatch):
    mock_random = MagicMock(return_value=0.67)
    monkeypatch.setattr("numpy.random.rand", mock_random)
    assert should_change_solution(32, 37, 100)
    mock_annealing_func.assert_called_once_with(32, 37, 100)
    mock_random.assert_called_once()


@patch("metaheuristics.simulated_annealing.simulated_annealing_utils.simulated_annealing_function",
       return_value=0.32)
def test_when_random_num_greater_than_func_val(mock_annealing_func, monkeypatch):
    mock_random = MagicMock(return_value=0.9)
    monkeypatch.setattr("numpy.random.rand", mock_random)
    assert not should_change_solution(29.8, 38.1, 1000)
    mock_annealing_func.assert_called_once_with(29.8, 38.1, 1000)
    mock_random.assert_called_once()
