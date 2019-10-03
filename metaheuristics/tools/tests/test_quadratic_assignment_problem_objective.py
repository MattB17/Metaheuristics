# -*- coding: utf-8 -*-

import pytest
import pandas as pd
from metaheuristics.tools.functions import quadratic_assignment_objective


@pytest.fixture(scope="module")
def mock_flow_matrix():
    facilities = ["X", "Y", "Z"]
    return pd.DataFrame([[1, 0, 1], [0, 1, 2], [2, 2, 1]],
                        columns=facilities, index=facilities)


@pytest.fixture(scope="module")
def mock_dist_matrix():
    facilities = ["X", "Y", "Z"]
    return pd.DataFrame([[0, 2, 0], [2, 0, 0], [0, 0, 0]],
                        columns=facilities, index=facilities)


def test_with_empty_dataframes():
    flow_matrix = pd.DataFrame()
    dist_matrix = pd.DataFrame()
    result = quadratic_assignment_objective(dist_matrix, flow_matrix, [])
    assert result == 0


def test_with_dataframe_having_one_element():
    flow_matrix = pd.DataFrame([[3]], columns=["A"], index=["A"])
    dist_matrix = pd.DataFrame([[2]], columns=["A"], index=["A"])
    result = quadratic_assignment_objective(dist_matrix, flow_matrix, ["A"])
    assert result == 6


def test_with_non_trivial_dataframes_no_reindexing(mock_flow_matrix, mock_dist_matrix):
    facility_order = ["X", "Y", "Z"]
    result = quadratic_assignment_objective(
            mock_dist_matrix, mock_flow_matrix, facility_order)
    assert result == 0


def test_with_non_trivial_dataframes_and_reindexing(mock_flow_matrix, mock_dist_matrix):
    facility_order = ["Z", "X", "Y"]
    result = quadratic_assignment_objective(
            mock_dist_matrix, mock_flow_matrix, facility_order)
    assert result == 8
