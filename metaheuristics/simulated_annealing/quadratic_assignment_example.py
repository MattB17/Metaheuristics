#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 10:31:57 2018

@author: matthewbuckley
"""

import pandas as pd
from metaheuristics.simulated_annealing.temperature_params import TemperatureParams
from metaheuristics.simulated_annealing.simulated_annealing_utils import quadratic_assignment_simulated_annealing_solver


initial_solution = ["B", "D", "A", "E", "C", "F", "G", "H"]


Dist = pd.DataFrame([[0,1,2,3,1,2,3,4],[1,0,1,2,2,1,2,3],[2,1,0,1,3,2,1,2],
                      [3,2,1,0,4,3,2,1],[1,2,3,4,0,1,2,3],[2,1,2,3,1,0,1,2],
                      [3,2,1,2,2,1,0,1],[4,3,2,1,3,2,1,0]],
                    columns=["A","B","C","D","E","F","G","H"],
                    index=["A","B","C","D","E","F","G","H"])

Flow = pd.DataFrame([[0,5,2,4,1,0,0,6],[5,0,3,0,2,2,2,0],[2,3,0,0,0,0,0,5],
                      [4,0,0,0,5,2,2,10],[1,2,0,5,0,10,0,0],[0,2,0,2,10,0,5,1],
                      [0,2,0,2,0,5,0,10],[6,0,5,10,0,1,10,0]],
                    columns=["A","B","C","D","E","F","G","H"],
                    index=["A","B","C","D","E","F","G","H"])

iteration_size = 20

def temperature_update_func(current_temp):
    return 0.9 * current_temp

temperature_params = TemperatureParams(initial_temperature=1500.0,
                                       temperature_changes=250,
                                       temperature_updater=temperature_update_func)


final_solution = quadratic_assignment_simulated_annealing_solver(
        Flow, Dist, initial_solution, iteration_size, temperature_params)  

print("The final solution is {0} with objective value {1}"
      .format(final_solution.get_solution_value(), final_solution.get_objective_value()))        
            