from metaheuristics.simulated_annealing.simulated_annealing_utils \
    import himmelblau_simulated_annealing_solver
from metaheuristics.simulated_annealing.temperature_params \
    import TemperatureParams

iteration_size = 15

def temperature_update_func(current_temp):
    return 0.85 * current_temp

temperature_params = TemperatureParams(initial_temperature=1000.0,
                                       temperature_changes=300,
                                       temperature_updater=temperature_update_func)

final_solution = himmelblau_simulated_annealing_solver(
        2, 1, 0.1, iteration_size, temperature_params)

print("The final solution is {0} with objective value {1}"
      .format(final_solution.get_solution_value(), final_solution.get_objective_value()))
