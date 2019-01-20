#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 21:47:09 2019

@author: matthewbuckley
"""

import numpy as np
import pandas as pd


def quadratic_assignment_problem_objective(dist_matrix, flow_matrix, facility_order):
    """Calculates the objective value for the quadratic assignment problem
    
    Parameters
    ----------
    dist_matrix: pandas.DataFrame
        The matrix of distance values. This is a square matrix where the 
        entry in row i, column j represents the distance from facility i
        to facility j
    flow_matrix: pandas.DataFrame
        The matrix of flow values. This is a square matrix where the 
        entry in row i, column j represents the flow from facility i
        to facility j
    facility_order: list
        A list representing an ordering of the facilities, corresponding
        to a solution
        
    Returns
    -------
    float
        The objective value for the solution given by facility_order to
        the quadratic assignment problem
        
    """
    dist_array = np.array(dist_matrix.reindex(columns=facility_order, index=facility_order))
    objective_value_array = np.array(pd.DataFrame(dist_array * flow_matrix))
    return sum(sum(objective_value_array))


def himmelblau(solution_tuple):
    """An implementation of the himmelblau function
    
    Parameters
    ----------
    solution_tuple: tuple
        A two element tuple with the x and y coordinates of a
        solution to the himmelblau function
        
    Returns
    -------
    float
        The value of the himmelblau function at the solution given
        by x and y
        
    """
    x, y = solution_tuple
    first_term = (x ** 2) + y - 11
    second_term = x + (y ** 2) - 7
    return (first_term ** 2) + (second_term ** 2)