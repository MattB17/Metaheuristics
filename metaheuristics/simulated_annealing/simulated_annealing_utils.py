import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from metaheuristics.simulated_annealing.temperature_params import TemperatureParams
import random
import copy
from metaheuristics.tools.functions \
    import quadratic_assignment_objective, himmelblau
from metaheuristics.tools.optimization_problem \
    import OptimizationProblem


def calculate_adjustment(additive_adjustment, multiplicative_adjustment, multiplicative_constant):
    """Calculates an adjustment to add to a coordinate of a solution
    to a continuous optimization problem

    Parameters
    ----------
    additive_adjustment: float
        A number used to decide if the adjustment will be negative
        or positive. If this number is less than 0.5, the adjustment
        is negative, otherwise it is possitive
    multiplicative_adjustment: float
        A number to help determine the magnitude of the adjustment
    multiplicative_constant: float
        A sensitivity parameter to help influence the magnitude of
        changes between solutions. A smaller number gives smaller changes

    Returns
    -------
    float
        The adjustment that will be applied to a coordinate of a solution
        to a continuous optimization problem

    """
    sign = 1 - (2 * (additive_adjustment < 0.5))
    return sign * multiplicative_constant * multiplicative_adjustment


def pick_neighbour_for_himmelblau(solution_tuple, multiplicative_constant):
    """Picks a solution in the neighbourhood of solution_tuple for
    the himmelblau function

    Parameters
    ----------
    solution_tuple: tuple
        A two element tuple containing the x an y coordinates of a
        solution to the himmelblau function
    multiplicative_constant: float
        A sensitivity parameter that determines the magnitude of the
        distance between solution_tuple and its neighbour. A smaller
        number implies a smaller distance between the two solutions

    Returns
    -------
    tuple
        A two element tuple containing the x and y coordinates of a
        solution to the himmelblau function in the neighbourhood of
        solution_tuple

    """
    x_additive_adjustment = np.random.rand()
    x_multiplicative_adjustment = np.random.rand()
    y_additive_adjustment = np.random.rand()
    y_multiplicative_adjustment = np.random.rand()
    return (solution_tuple[0] + calculate_adjustment(
            x_additive_adjustment, x_multiplicative_adjustment, multiplicative_constant),
            solution_tuple[1] + calculate_adjustment(
                    y_additive_adjustment, y_multiplicative_adjustment, multiplicative_constant))



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


def himmelblau_simulated_annealing_solver(initial_x, initial_y, multiplicative_constant, iteration_size, temp_params):
    """Runs the simulated annealing algorithm to minimize the himmelblau
    function, starting from the initial solution initial_x and initial_y

    Parameters
    ----------
    initial_x: float
        The x coordinate of an initial solution to the himmelblau function
    initial_y: float
        The y coordinate of an initial solution to the himmelblau function
    multiplicative_constant: float
        A sensitivity parameter used to move from one solution to another. A
        smaller value indicates that the two solutions are separated by a
        smaller distance
    iteration_size: function
        Determines the number of iterations at each temperature
    temp_params: TemperatureParams
        The TemperatureParams object specifying the temperature parameters needed
        for the simulated annealing algorithm

    Returns
    -------
    OptimizationSolution
        An OptimizationSolution object, representing the final solution found by
        the simualated annealing algorithm

    """
    neighbour_finder = lambda solution: pick_neighbour_for_himmelblau(solution, multiplicative_constant)
    himmelblau_problem = OptimizationProblem((initial_x, initial_y), himmelblau, neighbour_finder)
    iteration_size_function = lambda temperature: iteration_size
    temperatures, solutions = run_simulated_annealing(temp_params, iteration_size_function, himmelblau_problem)
    solution_vals = [solution.get_solution_value() for solution in solutions]
    plot_final_objective_value_per_temperature(temperatures, solution_vals, "Himmelblau Minimization")
    return solutions[-1]


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
        Determines the number of iterations at each temperature
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
    qap_problem = OptimizationProblem(initial_array, qap_obj_function, pick_neighbour_for_qap)
    iteration_size_function = lambda temperature: iteration_size
    temperatures, solutions = run_simulated_annealing(temp_params, iteration_size_function, qap_problem)
    solution_vals = [solution.get_solution_value() for solution in solutions]
    plot_final_objective_value_per_temperature(temperatures, solution_vals, "Quadratic Assignment Problem")
    return solutions[-1]


def run_simulated_annealing(temp_params, iteration_size, optimization_problem):
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
    optimization_problem: OptimizationProblem
        The optimization problem being solved by the algorithm

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
    for t in range(temp_params.get_number_of_temperatures()):
        for step in range(iteration_size(t)):
            new_solution = run_annealing_step(temp_params.get_current_temperature(), optimization_problem)
            optimization_problem.update_current_solution(new_solution)
        temperatures[t] = temp_params.get_current_temperature()
        final_solution_per_temperature[t] = optimization_problem.get_current_solution()
        temp_params.update_temperature()
    return temperatures, final_solution_per_temperature


def run_annealing_step(curr_temp, optimization_problem):
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
    optimization_problem: OptimizationProblem
        The optimization problem being solved by simulated annealing

    Returns
    -------
    OptimizationSolution
        The OptimizationSolution calculated in the current step

    """
    curr_solution = optimization_problem.get_current_solution()
    new_solution = optimization_problem.find_neighbour_solution()
    if should_change_solution(curr_solution.get_objective_value(), new_solution.get_objective_value(), curr_temp):
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
    if temperature == 0:
        return 0
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
