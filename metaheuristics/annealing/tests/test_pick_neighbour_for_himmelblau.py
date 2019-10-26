from unittest.mock import MagicMock, patch, call
from metaheuristics.annealing.utils import pick_neighbour_for_himmelblau


ADJUSTMENT_STR = "metaheuristics.annealing.utils.calculate_adjustment"


@patch(ADJUSTMENT_STR, side_effect=[0.4, 0.2])
def test_when_both_coordinates_increase(mock_adjuster, monkeypatch):
    mock_random = MagicMock(side_effect = [0.83, 0.8, 0.6, 0.4])
    monkeypatch.setattr("numpy.random.rand", mock_random)
    new_solution = pick_neighbour_for_himmelblau((5.1, 0.7), 0.5)
    assert round(new_solution[0], 2) == 5.50
    assert round(new_solution[1], 2) == 0.90
    calls = [call(0.83, 0.8, 0.5), call(0.6, 0.4, 0.5)]
    mock_adjuster.assert_has_calls(calls)
    assert mock_adjuster.call_count == 2

@patch(ADJUSTMENT_STR, side_effect=[-0.75, 0.35])
def test_when_coordinates_move_in_different_directions(mock_adjuster,
                                                       monkeypatch):
    mock_random = MagicMock(side_effect = [0.1, 0.75, 0.9, 0.35])
    monkeypatch.setattr("numpy.random.rand", mock_random)
    new_solution = pick_neighbour_for_himmelblau((2, 1.3), 1)
    assert round(new_solution[0], 2) == 1.25
    assert round(new_solution[1], 2) == 1.65
    calls = [call(0.1, 0.75, 1), call(0.9, 0.35, 1)]
    mock_adjuster.assert_has_calls(calls)
    assert mock_adjuster.call_count == 2


@patch(ADJUSTMENT_STR, side_effect=[-1.0, -0.3])
def test_when_both_coordinates_decrease(mock_adjuster, monkeypatch):
    mock_random = MagicMock(side_effect = [0.45, 0.5, 0.37, 0.15])
    monkeypatch.setattr("numpy.random.rand", mock_random)
    new_solution = pick_neighbour_for_himmelblau((2, 1.5), 2)
    assert round(new_solution[0], 2) == 1.00
    assert round(new_solution[1], 2) == 1.20
    calls = [call(0.45, 0.5, 2), call(0.37, 0.15, 2)]
    mock_adjuster.assert_has_calls(calls)
    assert mock_adjuster.call_count == 2
