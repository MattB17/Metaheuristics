#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 12:16:25 2019

@author: matthewbuckley
"""

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from temperature_params import TemperatureParams
from optimization_solution import OptimizationSolution
sys.path.append("../")
import random
import copy
from optimization_functions import quadratic_assignment_problem_objective


def pick_neighbour_for_qap(solution):
    """selects a neighbour of solution
    
    A neighbour is selected by swapping two randomly chosen elements
    of solution
    
    Parameters
    ----------
    solution: list
        A solution to the quadratic assignment problem, specifying an
        ordering of the facilities
        
    Returns
    -------
    list
        Another solution to the quadratic assignment problem, obtained
        by swapping two random facilities in the ordering given by solution
        
    """
    a, b = random.sample(range(0, len(solution)), 2)
    new_solution = copy.deepcopy(solution)
    new_solution[a] = solution[b]
    new_solution[b] = solution[a]
    return new_solution


def quadratic_assignment_simulated_annealing_solver(flow_matrix, dist_matrix, initial_array, iteration_size, temp_params):
    """Runs the simulated annealing algorithm for the quadratic assignment
    problem, starting from the initial solution given by initial_array
    
    Parameters
    ----------
    flow_matrix: pandas.DataFrame
        A square matrix, specifying the flows between facilities
    dist_matrix: pandas.DataFrame
        A square matrix, specifying the distances between the facilities
    inital_array: list
        An initial solution to the quadratic assignment problem, that is
        the list represents an ordering of the facilities
    iteration_size: function
        A function to determine the number of iterations of the algorithm
        for each temperature. The function accepts a float parameter,
        representing the current temperature, and returns an integer
        representing the number of iterations to take at the current temperature
    temp_params: TemperatureParams
        The TemperatureParams object specifying the temperature parameters needed
        for the simulated annealing algorithm
        
    Returns
    -------
    OptimizationSolution
        An OptimizationSolution object, representing the final solution found by
        the simualated annealing algorithm
        
    """
    qap_obj_function = lambda solution: quadratic_assignment_problem_objective(dist_matrix, flow_matrix, solution)
    initial_solution = OptimizationSolution(initial_array, qap_obj_function, pick_neighbour_for_qap)
    temperatures, solutions = run_simulated_annealing(temp_params, iteration_size, initial_solution)
    solution_vals = [solution.get_solution_value() for solution in solutions]
    plot_final_objective_value_per_temperature(temperatures, solution_vals, "Quadratic Assignment Problem")
    return solutions[-1]


def run_simulated_annealing(temp_params, iteration_size, initial_solution):
    """Runs the simulated annealing algorithm for a combinatorial problem
    
    Parameters
    ----------
    temp_params: TemperatureParams
        A TemperatureParams object for handling the temperature parameter
        throughout the simulated annealing algorithm
    iteration_size: function
        A function to calculate the number of steps taken at each temperature.
        The function accept one parameter, a float specifying the current
        temperature. It then returns an integer specifying the number of steps
        for the simulated annealing algorithm to take at that temperature
    initial_solution: OptimizationSolution
        The initial OptimizationSolution used to initialize the algorithm
        
    Returns
    -------
    list, list
        Two lists of equal size. The first list contains floats of the
        temperatures used in the algorithm. The second list contains
        OptimizationSolution objects specifying the final solution
        taken for each temperature
        
    """
    temperatures = [None for _ in range(temp_params.get_number_of_temperatures())]
    final_solution_per_temperature = [None for _ in range(temp_params.get_number_of_temperatures())]
    curr_solution = initial_solution
    for t in range(temp_params.get_number_of_temperatures()):
        for step in range(iteration_size(t)):
            curr_solution = run_annealing_step(temp_params.get_current_temperature(), curr_solution)
        temperatures[t] = temp_params.get_current_temperature()
        final_solution_per_temperature[t] = curr_solution
        temp_params.update_temperature()
    return temperatures, final_solution_per_temperature


def run_annealing_step(curr_temp, curr_solution):
    """Runs one step of the combinatorial version of the simulated
    annealing algorithm at the current temperature and solution
    
    This step generates a new solution in the neighbourhood of
    curr_solution. This new solution is taken if it has a better
    objective value or if the value of the simulated annealing
    function at the current iteration is greater than a random
    number in the range [0, 1]
    
    Parameters
    ----------
    curr_temp: float
        The current temperature at this step in the algorithm
    curr_solution: OptimizationSolution
        The current solution when this step is initiated
        
    Returns
    -------
    OptimizationSolution
        The OptimizationSolution calculated in the current step
        
    """
    new_solution = curr_solution.find_a_neighbour_solution()
    if should_change_solution(curr_solution.get_solution_value(), new_solution.get_solution_value(), curr_temp):
        return new_solution
    return curr_solution


def simulated_annealing_function(old_objective_value, new_objective_value, temperature):
    """Calculates the value of the simulated annealing function
    
    This function is used to decide if the algorithm should accept
    a worse solution in order to possibly find a better global optimum
    
    Parameters
    ----------
    old_objective_value: float
        The objective value of the current solution
    new_objective_value: float
        The objective value of the new candidate solution
    temperature: float
        The current temperature at that point in the algorithm.
        The temperature is a parameter denoting the willingness
        to accept worse solutions. Throughout the algorithm, the
        temperature decreases, causing the value of the simulated
        annealing function to decrease. This signifies a reduced 
        willingness to accept worse solutions as the algorithm progresses
        
    Returns
    -------
    float
        The value of the simulated annealing function based on the
        objective value of the current solution, the objective value
        of the candidate solution, and the current temperature
        
    """
    exponent = (new_objective_value - old_objective_value) / temperature
    return 1 / (np.exp(exponent))


def should_change_solution(old_objective_value, new_objective_value, temperature):
    """Determines whether the candidate solution should be accepted
    instead of the current solution.
    
    The candidate solution is accepted if it has a better objective
    value or if the value of the simulated annealing function is at
    least as large as a random number between 0 and 1
    
    Parameters
    ----------
    old_objective_value: float
        The objective value of the current solution
    new_objective_value: float
        The objective value of the new candidate solution
    temperature: float
        The current temperature at that point in the algorithm.
        The temperature is a parameter denoting the willingness
        to accept worse solutions. Throughout the algorithm, the
        temperature decreases, causing the value of the simulated
        annealing function to decrease. This signifies a reduced 
        willingness to accept worse solutions as the algorithm progresses
        
    Returns
    -------
    bool
        An indicator for whether the candidate solution should be
        accepted over the current solution. Returns True if the
        candidate solution has a better objective value than the
        current solution or if the value of the simulated annealing
        function is at least as large as a random number in [0, 1]
        
    """
    if new_objective_value < old_objective_value:
        return True
    random_number = np.random.rand()
    annealing_value = simulated_annealing_function(old_objective_value, new_objective_value, temperature)
    return random_number <= annealing_value


def plot_final_objective_value_per_temperature(temperatures, objective_values, problem_name):
    """Uses matplotlib to plot the final objective value for each 
    temperature used in the simulated annealing algorithm
    
    Parameters
    ----------
    temperatures: list
        The set of temperatures used in the algorithm
    objective_values: list
        The final objective value accepted by the simulated
        annealing algorithm for each temperature
    problem_name: str
        The name of the problem to which the simulated
        annuealing algorithm was applied
        
    Returns
    -------
    None
    
    Side Effect
    -----------
    Produces a plot of objective_values versus temperatures
    
    """
    max_temp = max(temperatures)
    min_temp = min(temperatures)
    plt.plot(temperatures, objective_values)
    plt.title("Final Cost by Temperature for {} Problem".format(problem_name), fontsize=20, fontweight="bold")
    plt.xlabel("Temperatures", fontsize=18, fontweight="bold")
    plt.ylabel("Final Cost", fontsize=18, fontweight="bold")
    plt.xlim(max_temp, min_temp)
    plt.xticks(np.arange(min_temp, max_temp, 100), fontweight="bold")
    plt.yticks(fontweight="bold")
    plt.show()
