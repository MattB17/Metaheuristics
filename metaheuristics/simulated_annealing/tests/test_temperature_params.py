#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 21:11:49 2019

@author: matthewbuckley
"""

from metaheuristics.simulated_annealing.temperature_params import TemperatureParams
from unittest.mock import call, MagicMock


def test_can_create():
    initial_temp = 1250
    temp_changes = 150
    temp_updater = MagicMock()
    temp_params = TemperatureParams(initial_temp, temp_changes, temp_updater)
    assert temp_params.get_initial_temperature() == initial_temp
    assert temp_params.get_current_temperature() == initial_temp
    assert temp_params.get_number_of_temperatures() == temp_changes
    temp_updater.assert_not_called()
    
    
def test_temp_updater_called_once():
    initial_temp = 2000
    temp_changes = 100
    temp_updater = MagicMock(return_value=1900)
    temp_params = TemperatureParams(initial_temp, temp_changes, temp_updater)
    assert temp_params.get_initial_temperature() == initial_temp
    assert temp_params.get_current_temperature() == initial_temp
    temp_params.update_temperature()
    assert temp_params.get_initial_temperature() == initial_temp
    assert temp_params.get_current_temperature() == 1900
    temp_updater.assert_called_once_with(2000)


def test_temp_updater_called_multiple_times():
    initial_temp = 390
    temp_changes = 50
    temp_updater = MagicMock(side_effect=[380, 370, 360])
    temp_params = TemperatureParams(initial_temp, temp_changes, temp_updater)
    assert temp_params.get_initial_temperature() == initial_temp
    assert temp_params.get_current_temperature() == initial_temp
    temp_params.update_temperature()
    assert temp_params.get_initial_temperature() == initial_temp
    assert temp_params.get_current_temperature() == 380
    temp_params.update_temperature()
    assert temp_params.get_initial_temperature() == initial_temp
    assert temp_params.get_current_temperature() == 370
    temp_params.update_temperature()
    assert temp_params.get_initial_temperature() == initial_temp
    assert temp_params.get_current_temperature() == 360
    updater_calls = [call(390), call(380), call(370)]
    assert temp_updater.call_count == 3
    temp_updater.assert_has_calls(updater_calls)

    
    