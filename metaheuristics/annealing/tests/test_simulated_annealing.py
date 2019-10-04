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


def test_can_instantiate(mock_temps, mock_iter_func, mock_problem):
    annealing = SimulatedAnnealing(mock_temps, mock_iter_func, mock_problem)
    assert annealing.get_temp_params() == mock_temps
    assert annealing.get_iteration_function() == mock_iter_func
    assert annealing.get_optimization_problem() == mock_problem
