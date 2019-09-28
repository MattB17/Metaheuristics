import numpy as np


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
    reindexed_dists = dist_matrix.reindex(columns=facility_order, index=facility_order)
    #convert reindexed_dists to array, otherwise indices are matched when multiplying
    objective_value_df = np.array(reindexed_dists) * flow_matrix
    if objective_value_df.empty:
        return 0
    return sum(sum(np.array(objective_value_df)))


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
