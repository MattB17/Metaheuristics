from unittest.mock import patch, call
from metaheuristics.annealing.utils import himmelblau_neighbour_finder


NEIGHBOUR_STR = "metaheuristics.annealing.utils.pick_neighbour_for_himmelblau"


@patch(NEIGHBOUR_STR, return_value=(2,1))
def test_with_small_constant(mock_neighbour_picker):
    neighbour_finder = himmelblau_neighbour_finder(0.5)
    initial_solution = (1.2, 0.7)
    assert neighbour_finder(initial_solution) == (2,1)
    mock_neighbour_picker.assert_called_once_with(initial_solution, 0.5)

@patch(NEIGHBOUR_STR, return_value=(-0.7, 1.4))
def test_with_large_constant(mock_neighbour_picker):
    neighbour_finder = himmelblau_neighbour_finder(2.2)
    initial_solution = (-2.5, 1.9)
    assert neighbour_finder(initial_solution) == (-0.7, 1.4)
    mock_neighbour_picker.assert_called_once_with(initial_solution, 2.2)
