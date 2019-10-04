"""A set of methods and classes used to implement the simulated annealing
metaheuristics algorithm. Simulating annealing is a probabilistic metaheuristic
algorithm under the general class of descent methods.

For a minimization problem, one can think of the solution space as a hilly
region where valleys correspond to local optima. The global optima corresponds
to the lowest valley.

In general descent methods, given a solution, solvers try to move towards the
closest valley in order to minimize a localized version of the problem. However,
it is possible that this local optima is far greater than the global optima,
leading to a sub optimal solution.

Simulated Annealing also takes this approach but with the added feature that
some uphill moves are allowed. The uphill moves are used to get out of local
optima in the hopes of finding another region in which the local optima is also
a global optima. Uphill moves are controlled by a temperature parameter which
decreases over time, corresponding to a reduced probability of accepting uphill
moves. The temperature is a parameter denoting the willingness to accept worse
solutions. Throughout the algorithm, the temperature decreases, causing the
value of the simulated annealing function to decrease.

"""
from metaheuristics.annealing.temp_params import TempParams
