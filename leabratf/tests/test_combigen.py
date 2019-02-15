"""
Tests for leabratf.utils.py
"""
import logging

import pytest
import numpy as np

from leabratf.tasks.combinatorics import combigen

test_y = [np.array([[0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [1, 1]]),]
test_x = [np.array([[1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1]]),]
test_arrays = list(zip(test_x, test_y))

@pytest.mark.parametrize("shapes", [(1,5,3), (2,2,2)])
def test_generate_labels_returns_correct_shapes(shapes):
    assert np.array_equal(combigen.generate_labels(*shapes).shape, shapes)

@pytest.mark.parametrize("test_x,test_y", test_arrays)
def test_inverse_transform_single_sample(test_x, test_y):
    gen_x = combigen.inverse_transform_single_sample(test_y)
    assert np.array_equal(test_x, gen_x)

def test_inverse_transform():
    gen_X = combigen.inverse_transform(test_y)
    assert np.array_equal(test_x, gen_X)
        
