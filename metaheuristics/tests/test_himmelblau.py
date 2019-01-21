# -*- coding: utf-8 -*-
from metaheuristics.optimization_functions import himmelblau


def test_with_both_coordinates_zero():
    result = himmelblau((0, 0))
    assert round(result, 2) == 170.00
    
    
def test_with_both_coordinates_negative():
    result = himmelblau((-1.5, -2.4))
    assert round(result, 2) == 131.83


def test_with_both_coordinates_positive():
    result = himmelblau((2.0, 0.5))
    assert round(result, 2) == 64.81
    

def test_with_x_negative_y_positive():
    result = himmelblau((-3, 1))
    assert round(result, 2) == 82.00
    

def test_with_x_positive_y_negative():
    result = himmelblau((2, -2))
    assert round(result, 2) == 82.00    
