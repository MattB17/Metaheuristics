import pytest
from unittest.mock import MagicMock, call
from metaheuristics.annealing.simulated_annealing import SimulatedAnnealing


@pytest.fixture(scope="function")
def mock_temps():
    temps = MagicMock()
    temps.update_temp = MagicMock(side_effect=None)
    return temps


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


def test_when_solution_changes(annealing, mock_problem):
    curr_solution = MagicMock()
    new_solution = MagicMock()
    mock_problem.get_current_solution = MagicMock(return_value=curr_solution)
    mock_problem.find_neighbour_solution = MagicMock(return_value=new_solution)
    curr_solution.get_objective_value = MagicMock(return_value=12)
    new_solution.get_objective_value = MagicMock(return_value=9)
    annealing.should_change_solution = MagicMock(return_value=True)
    assert annealing.run_annealing_step() == new_solution
    mock_problem.get_current_solution.assert_called_once()
    mock_problem.find_neighbour_solution.assert_called_once()
    annealing.should_change_solution.assert_called_once_with(12, 9)


def test_when_solution_does_not_change(annealing, mock_problem):
    curr_solution = MagicMock()
    new_solution = MagicMock()
    mock_problem.get_current_solution = MagicMock(return_value=curr_solution)
    mock_problem.find_neighbour_solution = MagicMock(return_value=new_solution)
    curr_solution.get_objective_value = MagicMock(return_value=17.5)
    new_solution.get_objective_value = MagicMock(return_value=37)
    annealing.should_change_solution = MagicMock(return_value=False)
    assert annealing.run_annealing_step() == curr_solution
    mock_problem.get_current_solution.assert_called_once()
    mock_problem.find_neighbour_solution.assert_called_once()
    annealing.should_change_solution.assert_called_once_with(17.5, 37)


def test_annealing_iteration_no_steps(annealing, mock_temps,
                                      mock_iter_func, mock_problem):
    mock_iter_func.return_value = 0
    annealing.run_annealing_step = MagicMock()
    mock_problem.update_current_solution = MagicMock()
    mock_temps.get_current_temp = MagicMock(return_value=2.5)
    solution1 = MagicMock()
    solution2 = MagicMock()
    mock_problem.get_current_solution = MagicMock(return_value=solution2)
    temps = [4.8, 3.9, None, None, None]
    solutions = [solution1, solution2, None, None, None]
    annealing.perform_annealing_iteration(2, temps, solutions)
    mock_iter_func.assert_called_once_with(2)
    annealing.run_annealing_step.assert_not_called()
    mock_problem.update_current_solution.assert_not_called()
    mock_temps.get_current_temp.assert_called_once()
    mock_problem.get_current_solution.assert_called_once()
    mock_temps.update_temp.assert_called_once()
    assert temps == [4.8, 3.9, 2.5, None, None]
    assert solutions == [solution1, solution2, solution2, None, None]


def test_annealing_iteration_one_step(annealing, mock_temps,
                                      mock_iter_func, mock_problem):
    mock_iter_func.return_value = 1
    solution = MagicMock()
    annealing.run_annealing_step = MagicMock(return_value=solution)
    mock_problem.update_current_solution = MagicMock(side_effect=None)
    mock_temps.get_current_temp = MagicMock(return_value=10)
    mock_problem.get_current_solution = MagicMock(return_value=solution)
    temps = [None, None]
    solutions = [None, None]
    annealing.perform_annealing_iteration(0, temps, solutions)
    mock_iter_func.assert_called_once_with(0)
    annealing.run_annealing_step.assert_called_once()
    mock_problem.update_current_solution.assert_called_once_with(solution)
    mock_temps.get_current_temp.assert_called_once()
    mock_problem.get_current_solution.assert_called_once()
    mock_temps.update_temp.assert_called_once()
    assert temps == [10, None]
    assert solutions == [solution, None]


def test_annealing_iteration_multi_step(annealing, mock_temps,
                                        mock_iter_func, mock_problem):
    mock_iter_func.return_value = 3
    solutions = [MagicMock(), MagicMock(), MagicMock(),
                 MagicMock(), MagicMock()]
    annealing.run_annealing_step = MagicMock(
        side_effect=[solutions[3], solutions[4], solutions[4]])
    mock_problem.update_current_solution = MagicMock(side_effect=None)
    mock_temps.get_current_temp = MagicMock(return_value=0.5)
    mock_problem.get_current_solution = MagicMock(return_value=solutions[4])
    temps = [2, 1.5, 1, None]
    solutions_list = [solutions[0], solutions[1], solutions[2], None]
    annealing.perform_annealing_iteration(3, temps, solutions_list)
    mock_iter_func.assert_called_once_with(3)
    assert annealing.run_annealing_step.call_count == 3
    update_calls = [call(solutions[3]), call(solutions[4]), call(solutions[4])]
    mock_problem.update_current_solution.assert_has_calls(update_calls)
    assert mock_problem.update_current_solution.call_count == 3
    mock_temps.get_current_temp.assert_called_once()
    mock_problem.get_current_solution.assert_called_once()
    mock_temps.update_temp.assert_called_once()
    assert temps == [2, 1.5, 1, 0.5]
    assert solutions_list == [solutions[0], solutions[1],
                              solutions[2], solutions[4]]
