"""Tests for the combigen task."""
import logging

import pytest
import numpy as np

from leabratf.tasks.combinatorics import combigen

test_y = [np.array([[0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [1, 1]])]
test_x = [np.array([[1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1]]),]
test_arrays = list(zip(test_x, test_y))

@pytest.mark.parametrize("shapes", [(1,5,3), (2,2,2)])
def test_generate_labels_returns_correct_shapes(shapes):
    assert np.array_equal(combigen.generate_labels(*shapes).shape, shapes)

def test_generate_labels_doesnt_return_all_true():
    # Generate a large number of y values to test
    large_test_Y = combigen.generate_labels(1000000)

    # Sum over the long dimension of each sample to see how many of them are
    # set to the on state. If they are all on, then it will sum to the
    # length of the dim.
    values_in_sum = np.isin(range(large_test_Y.shape[1] + 1),
                            np.sum(large_test_Y, axis=1))
    assert all(values_in_sum == large_test_Y.shape[1]*[True] + [False])    

@pytest.mark.parametrize("test_x,test_y", test_arrays)
def test_inverse_transform_single_sample(test_x, test_y):
    gen_x = combigen.inverse_transform_single_sample(test_y)
    assert np.array_equal(test_x, gen_x)

def test_inverse_transform():
    gen_X = combigen.inverse_transform(np.array(test_y))
    assert np.array_equal(test_x, gen_X)        
