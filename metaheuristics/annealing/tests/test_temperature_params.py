import pytest
from metaheuristics.annealing.temp_params import TempParams
from unittest.mock import call, MagicMock


@pytest.fixture(scope="function")
def updater():
    return MagicMock()


def test_can_create(updater):
    initial_temp = 1250
    temp_changes = 150
    temp_params = TempParams(initial_temp, temp_changes, updater)
    assert temp_params.get_initial_temp() == initial_temp
    assert temp_params.get_current_temp() == initial_temp
    assert temp_params.get_number_of_temps() == temp_changes
    updater.assert_not_called()


def test_temp_updater_called_once():
    initial_temp = 2000
    temp_changes = 100
    temp_updater = MagicMock(return_value=1900)
    temp_params = TempParams(initial_temp, temp_changes, temp_updater)
    assert temp_params.get_initial_temp() == initial_temp
    assert temp_params.get_current_temp() == initial_temp
    temp_params.update_temp()
    assert temp_params.get_initial_temp() == initial_temp
    assert temp_params.get_current_temp() == 1900
    temp_updater.assert_called_once_with(2000)


def test_temp_updater_called_multiple_times():
    initial_temp = 390
    temp_changes = 50
    temp_updater = MagicMock(side_effect=[380, 370, 360])
    temp_params = TempParams(initial_temp, temp_changes, temp_updater)
    assert temp_params.get_initial_temp() == initial_temp
    assert temp_params.get_current_temp() == initial_temp
    temp_params.update_temp()
    assert temp_params.get_initial_temp() == initial_temp
    assert temp_params.get_current_temp() == 380
    temp_params.update_temp()
    assert temp_params.get_initial_temp() == initial_temp
    assert temp_params.get_current_temp() == 370
    temp_params.update_temp()
    assert temp_params.get_initial_temp() == initial_temp
    assert temp_params.get_current_temp() == 360
    updater_calls = [call(390), call(380), call(370)]
    assert temp_updater.call_count == 3
    temp_updater.assert_has_calls(updater_calls)


def test_equal_without_updates(updater):
    temp_val = 225
    changes = 100
    assert (TempParams(temp_val, changes, updater)
            == TempParams(temp_val, changes, updater))


def test_equal_after_updates():
    temp_val = 225
    changes = 100
    updater = MagicMock(side_effect=[220, 215, 220, 215])
    temps1 = TempParams(temp_val, changes, updater)
    temps2 = TempParams(temp_val, changes, updater)
    temps1.update_temp()
    temps1.update_temp()
    temps2.update_temp()
    temps2.update_temp()
    assert temps1 == temps2


def test_not_equal_with_different_initial_temps(updater):
    temp1 = 145
    temp2 = 150
    changes = 15
    assert (TempParams(temp1, changes, updater)
            != TempParams(temp2, changes, updater))


def test_different_change_count_gives_not_equal(updater):
    temp = 170.7
    changes1 = 70
    changes2 = 67
    assert (TempParams(temp, changes1, updater)
            != TempParams(temp, changes2, updater))


def test_different_updaters_gives_not_equal(updater):
    temp = 12.5
    changes = 30
    updater2 = MagicMock()
    assert (TempParams(temp, changes, updater)
            != TempParams(temp, changes, updater2))


def test_different_current_temp_gives_not_equal():
    temp = 12.5
    changes = 30
    updater = MagicMock(side_effect=[12.4, 12.3, 12.4])
    temps1 = TempParams(temp, changes, updater)
    temps2 = TempParams(temp, changes, updater)
    temps1.update_temp()
    temps1.update_temp()
    temps2.update_temp()
    assert temps1 != temps2
