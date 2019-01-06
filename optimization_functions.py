#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 21:47:09 2019

@author: matthewbuckley
"""


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


def himmelblau(x, y):
    """An implementation of the himmelblau function
    
    Parameters
    ----------
    x: float
        The x coordinate of the solution
    y: float
        The y coordinate of the solution
        
    Returns
    -------
    float
        The value of the himmelblau function at the solution given
        by x and y
        
    """
    first_term = (x ** 2) + y - 11
    second_term = x + (y ** 2) - 7
    return (first_term ** 2) + (second_term ** 2)