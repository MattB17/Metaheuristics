# -*- coding: utf-8 -*-

from metaheuristics.simulated_annealing.simulated_annealing_utils \
    import simulated_annealing_function


def test_with_zero_temperature():
    assert simulated_annealing_function(20, 30, 0) == 0


def test_with_same_objective_value():
    assert simulated_annealing_function(15.7, 15.7, 1000) == 1


def test_with_worse_objective_value():
    result = simulated_annealing_function(11.2, 12.7, 100)
    assert round(result, 2) == 0.99


def test_with_better_objective_value():
    result = simulated_annealing_function(10.5, 1.5, 100)
    assert round(result, 2) == 1.09


def test_with_small_temperature():
    result = simulated_annealing_function(11.2, 12.7, 10)
    assert round(result, 2) == 0.86


def test_with_large_temperature():
    result = simulated_annealing_function(50, 100, 1000)
    assert round(result, 2) == 0.95
