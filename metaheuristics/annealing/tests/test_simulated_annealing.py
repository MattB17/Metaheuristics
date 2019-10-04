import pytest
from unittest.mock import MagicMock
from metaheuristics.annealing.simulated_annealing import SimulatedAnnealing


@pytest.fixture(scope="function")
def mock_temps():
    return MagicMock()


@pytest.fixture(scope="function")
def mock_iter_func():
    return MagicMock()


@pytest.fixture(scope="function")
def mock_problem():
    return MagicMock()


@pytest.fixture(scope="function")
def annealing(mock_temps, mock_iter_func, mock_problem):
    return SimulatedAnnealing(mock_temps, mock_iter_func, mock_problem)


def test_can_instantiate(mock_temps, mock_iter_func, mock_problem, annealing):
    assert annealing.get_temp_params() == mock_temps
    assert annealing.get_iteration_function() == mock_iter_func
    assert annealing.get_optimization_problem() == mock_problem


def test_function_with_zero_temperature(annealing, mock_temps):
    mock_temps.get_current_temp = MagicMock(return_value=0)
    assert annealing.annealing_function(20, 30) == 0
    mock_temps.get_current_temp.assert_called_once()


def test_function_with_same_objective_value(annealing, mock_temps):
    mock_temps.get_current_temp = MagicMock(return_value=1000)
    assert annealing.annealing_function(15.7, 15.7) == 1
    assert mock_temps.get_current_temp.call_count == 2


def test_function_with_worse_objective_value(annealing, mock_temps):
    mock_temps.get_current_temp = MagicMock(return_value=100)
    result = annealing.annealing_function(11.2, 12.7)
    assert round(result, 2) == 0.99
    assert mock_temps.get_current_temp.call_count == 2


def test_function_with_better_objective_value(annealing, mock_temps):
    mock_temps.get_current_temp = MagicMock(return_value=100)
    result = annealing.annealing_function(10.5, 1.5)
    assert round(result, 2) == 1.09
    assert mock_temps.get_current_temp.call_count == 2


def test_function_with_small_temperature(annealing, mock_temps):
    mock_temps.get_current_temp = MagicMock(return_value=10)
    result = annealing.annealing_function(11.2, 12.7)
    assert round(result, 2) == 0.86
    assert mock_temps.get_current_temp.call_count == 2


def test_function_with_large_temperature(annealing, mock_temps):
    mock_temps.get_current_temp = MagicMock(return_value=1000)
    result = annealing.annealing_function(50, 100)
    assert round(result, 2) == 0.95
    assert mock_temps.get_current_temp.call_count == 2


def test_should_change_with_better_objective_value(annealing, monkeypatch):
    mock_random = MagicMock()
    monkeypatch.setattr("numpy.random.rand", mock_random)
    annealing.annealing_function = MagicMock()
    assert annealing.should_change_solution(25, 20)
    annealing.annealing_function.assert_not_called()
    mock_random.assert_not_called()


def test_should_change_when_random_num_less_than_func_val(annealing,
                                                          monkeypatch):
    mock_random = MagicMock(return_value=0.5)
    monkeypatch.setattr("numpy.random.rand", mock_random)
    annealing.annealing_function = MagicMock(return_value=0.75)
    assert annealing.should_change_solution(10, 12.5)
    annealing.annealing_function.assert_called_once_with(10, 12.5)
    mock_random.assert_called_once()


def test_should_change_when_random_num_equals_func_val(annealing, monkeypatch):
    mock_random = MagicMock(return_value=0.67)
    monkeypatch.setattr("numpy.random.rand", mock_random)
    annealing.annealing_function = MagicMock(return_value=0.67)
    assert annealing.should_change_solution(32, 37)
    annealing.annealing_function.assert_called_once_with(32, 37)
    mock_random.assert_called_once()


def test_should_not_change_when_random_num_greater_than_func_val(annealing,
                                                                 monkeypatch):
    mock_random = MagicMock(return_value=0.9)
    monkeypatch.setattr("numpy.random.rand", mock_random)
    annealing.annealing_function = MagicMock(return_value=0.32)
    assert not annealing.should_change_solution(29.8, 38.1)
    annealing.annealing_function.assert_called_once_with(29.8, 38.1)
    mock_random.assert_called_once()
