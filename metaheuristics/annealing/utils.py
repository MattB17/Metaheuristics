"""A set of methods used in running simulated annealing
"""
import numpy as np


def calculate_adjustment(adjustment_indicator, multiplicative_adjustment,
                         multiplicative_constant):
    """An adjustment to a solution coordinate to a continuous problem

    Parameters
    ----------
    adjustment_indicator: float
        A number used to decide if the adjustment will be negative
        or positive. If this number is less than 0.5, the adjustment
        is negative, otherwise it is positive
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
    sign = 1 - (2 * (adjustment_indicator < 0.5))
    return sign * multiplicative_constant * multiplicative_adjustment


def pick_neighbour_for_himmelblau(solution_tuple, multiplicative_constant):
    """Picks a neighbour of `solution_tuple` for the himmelblau function

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
    x_adjustment_indicator = np.random.rand()
    x_multiplic_adjustment = np.random.rand()
    y_adjustment_indicator = np.random.rand()
    y_multiplic_adjustment = np.random.rand()
    return (solution_tuple[0] + calculate_adjustment(x_adjustment_indicator,
                                                     x_multiplic_adjustment,
                                                     multiplicative_constant),
            solution_tuple[1] + calculate_adjustment(y_adjustment_indicator,
                                                     y_multiplic_adjustment,
                                                     multiplicative_constant))


def himmelblau_neighbour_finder(multiplicative_constant):
    """A function to move between neighbour solutions for himmelblau

    Parameters
    ----------
    multiplicative_constant: float
        A sensitivity parameter used to move from one solution to another. A
        smaller value indicates that the two solutions are separated by a
        smaller distance

    Returns
    -------
    function
        A function for moving between neighbouring solutions for the himmelblau
        problem. The function takes one argument, a two element tuple specifying
        a current solution to the himmelblau problem. The function returns a
        two element tuple specifying a neighbouring solution

    """
    return lambda solution: pick_neighbour_for_himmelblau(
        solution, multiplicative_constant)
