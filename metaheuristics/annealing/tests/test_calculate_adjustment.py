from metaheuristics.annealing.utils import calculate_adjustment


def test_with_negative_sign():
    assert round(calculate_adjustment(0.45, 10, 0.1), 2) == -1.00


def test_with_positive_sign():
    assert round(calculate_adjustment(0.7, 3, 0.5), 2) == 1.50
