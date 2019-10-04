"""The TempParams class is a container class for tempeature parameters used
in the simulated annealing algorithm. The class keeps track of how the initial
and current temperature, the number of temperature changes, and the function
used to move from one temperature to the next

"""


class TempParams:
    """A container class to handle parameters related to the temperature
    used during the simulated annealing algorithm

    Attributes
    ----------
    initial_temperature: float
        The temperature used at the beginning of the algorithm
    temperature_changes: int
        The number of times the temperature should change during the algorithm
    temperature_updater: function
        A function used to move between temperatures during the algorithm. The
        function accepts a single integer parameter, representing the current
        temperature, and returns an integer, representing the new temperature
    current_temperature: float
        The temperature at a particular point during the algorithm

    """
    def __init__(self, initial_temp, temp_changes, temp_updater):
        self._initial_temp = initial_temp
        self._temp_changes = temp_changes
        self._temp_updater = temp_updater
        self._current_temp = initial_temp

    def get_initial_temp(self):
        """Retrieves the initial temperature used in the algorithm

        Returns
        -------
        float
            The initial temperature used in the algorithm

        """
        return self._initial_temp

    def get_number_of_temps(self):
        """Retrieves the number of times the temperature changes
        during the simulated annealing algorithm

        Returns
        -------
        int
            The number of temperature changes during the algorithm

        """
        return self._temp_changes

    def update_temp(self):
        """Updates the current temperature using temperature_updater

        Returns
        -------
        None

        Side Effect
        -----------
        changes the current_temperature attribute by applying
        temperature_updater to it

        """
        self._current_temp = self._temp_updater(self._current_temp)

    def get_current_temp(self):
        """Retrieves the current temperature at a point in the algorithm

        Returns
        -------
        float
            The value of the temperature parameter at a certain point
            during the simulated annealing algorithm

        """
        return self._current_temp

    def __eq__(self, other_params):
        """Checks if two `TempParams` objects are equal

        Two `TempParams` are deemed equal if the have the same initial
        temperature, the same current temperature, the same number of
        temperature changes, and the same function used to update the current
        temperature to the next

        Parameters
        ----------
        other_params: TempParams
            The second `TempParams` object being compared to self

        Returns
        -------
        bool
            True if the two `TempParams` objects are equal, otherwise False

        """
        same_initial = self._initial_temp == other_params._initial_temp
        same_changes = self._temp_changes == other_params._temp_changes
        same_updater = self._temp_updater == other_params._temp_updater
        same_current = self._current_temp == other_params._current_temp
        return same_initial and same_changes and same_updater and same_current
