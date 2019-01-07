#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 22:48:00 2019

@author: matthewbuckley
"""

class TemperatureParams:
    
    def __init__(self, initial_temperature, temperature_changes, temperature_updater):
        self._initial_temperature = initial_temperature
        self._temperature_changes = temperature_changes
        self._temperature_updater = temperature_updater
        self._current_temperature = initial_temperature
        
    def get_initial_temperature(self):
        return self._initial_temperature
    
    def get_number_of_temperatures(self):
        return self._temperature_changes
    
    def update_temperature(self):
        self._current_temperature = self._temperature_updater(self._current_temperature)
        
    def get_current_temperature(self):
        return self._current_temperature
