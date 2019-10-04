"""The SimulatedAnnealing class is used to run the simulated annealing
algorithm in order to solve optimization problems

"""
from metaheuristics.annealing.temp_params import TempParams
from metaheuristics.tools.optimization_problem import OptimizationProblem


class SimulatedAnnealing:
    """To run the simulated annealing algorithm on an optimization problem

    Parameters
    ----------
    temp_params: TempParams
        A `TempParams` object storing the temperature information for the
        algorithm
    iteration_function: function
        A function to calculate the number of steps taken at each temperature.
        The function accepts one parameter, a float specifying the current
        temperature. It then returns an integer specifying the number of steps
        for the simulated annealing algorithm to take at that temperature
    problem: OptimizationProblem
        The `OptimizationProblem` being solved by the algorithm

    Attributes
    ----------
    _temp_params: TempParams
        A `TempParams` object for the temperatures used by the algorithm
    _iteration_function: function
        A function calculating iteration size for a given temperature
    problem: OptimizationProblem
        The problem being solved

    """
    def __init__(self, temp_params, iteration_function, problem):
        self._temp_params = temp_params
        self._iteration_function = iteration_function
        self._problem = problem

    def get_temp_params(self):
        """The temperature information used by the algorithm

        Returns
        -------
        TempParams
            The `TempParams` object storing temperature information for the
            algorithm

        """
        return self._temp_params

    def get_iteration_function(self):
        """The function to calculate iteration size at each temperature

        Returns
        -------
        function
            The function used by the algorithm to determine the number of
            iterations to perform for a given temperature

        """
        return self._iteration_function

    def get_optimization_problem(self):
        """The problem being solved by the algorithm

        Returns
        -------
        OptimizationProblem
            The `OptimizationProblem` object representing the problem being
            solved by the algorithm

        """
        return self._problem
