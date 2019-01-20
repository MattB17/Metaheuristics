#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 21:51:17 2019

@author: matthewbuckley
"""

class OptimizationSolution:
    """A container class for a solution to an optimization problem
    
    Attributes
    ----------
    solution_value: object
        An object representing the solution itself. For example, in
        the QAP problem this would be a list, specifying an ordering
        of the facilities, whereas for the himmelblau problem this
        would be a tuple specifying the x and y coordinates
    objective_value: float
        The objective value of the solution based on the optimization
        problem
        
    """
    def __init__(self, solution_value, objective_value):
        self._solution_value = solution_value
        self._objective_value = objective_value
        
    def get_solution_value(self):
        """Retrieves the solution for an optimization problem
        
        Returns
        -------
        object
            The solution to an optimization problem
            
        """
        return self._solution_value
    
    def get_objective_value(self):
        """Retrieves the objective value of the solution
        
        Returns
        -------
        float
            The objective value of the solution, based on an
            optimization problem
            
        """
        return self._objective_value