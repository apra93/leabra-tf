"""Tests for the visualization function for the combigen task."""
import logging

import pytest
import numpy as np

from leabratf.visualization import combigen_heatmap as ch

N = [1, 3, 10]

@pytest.mark.parametrize("length", N)
def test_combigen_heatmap_runs_without_errors_for_different_input_lengths(
        length):
    """Lifted directly from nb0.1 c382."""
    ch.heatmap(np.random.choice(2, (length,5,2), True))

@pytest.mark.parametrize("n_pairs", N)
def test_visualize_combigen_runs_without_errors_for_different_numbers_of_pairs(
        n_pairs):
    ch.visualize_combigen(n_pairs)
