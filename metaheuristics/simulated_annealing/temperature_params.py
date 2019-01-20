#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 22:48:00 2019

@author: matthewbuckley
"""

class TemperatureParams:
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
    def __init__(self, initial_temperature, temperature_changes, temperature_updater):
        self._initial_temperature = initial_temperature
        self._temperature_changes = temperature_changes
        self._temperature_updater = temperature_updater
        self._current_temperature = initial_temperature
        
    def get_initial_temperature(self):
        """Retrieves the initial temperature used in the algorithm
        
        Returns
        -------
        float
            The initial temperature used in the algorithm
        
        """
        return self._initial_temperature
    
    def get_number_of_temperatures(self):
        """Retrieves the number of times the temperature changes
        during the simulated annealing algorithm
        
        Returns
        -------
        int
            The number of temperature changes during the algorithm
        
        """
        return self._temperature_changes
    
    def update_temperature(self):
        """Updates the current temperature using temperature_updater
        
        Returns
        -------
        None
        
        Side Effect
        -----------
        changes the current_temperature attribute by applying
        temperature_updater to it
        
        """
        self._current_temperature = self._temperature_updater(self._current_temperature)
        
    def get_current_temperature(self):
        """Retrieves the current temperature at a point in the algorithm
        
        Returns
        -------
        float
            The value of the temperature parameter at a certain point
            during the simulated annealing algorithm
            
        """
        return self._current_temperature
