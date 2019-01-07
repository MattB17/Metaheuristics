#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 22:22:43 2019

@author: matthewbuckley
"""

class OptimizationSolution:
    
    def __init__(self, solution, objective_function, solution_updater):
        self._solution = solution
        self._objective_function = objective_function
        self._solution_updater = solution_updater
        self._solution_value = None
        
    def get_solution(self):
        return self._solution
    
    def get_objective_function(self):
        return self._objective_function
    
    def get_solution_value(self):
        if self._solution_value is None:
            self._solution_value = self._objective_function(self._solution)
        return self._solution_value
    
    def find_a_neighbour_solution(self):
        return self._solution_updater(self._solution)
        