"""Tests for the combigen task."""
import logging

import pytest
import numpy as np

from leabratf.tasks.combinatorics import combigen

test_y = np.array([[[[1, 1],
                     [0, 1],
                     [0, 0],
                     [0, 0],
                     [0, 0]],
                    [[1, 1],
                     [0, 0],
                     [0, 1],
                     [0, 0],
                     [0, 1]],
                    [[0, 1],
                     [1, 0],
                     [1, 0],
                     [1, 1],
                     [0, 0]],
                    [[0, 1],
                     [0, 0],
                     [0, 1],
                     [1, 0],
                     [0, 0]]]])

test_x = np.array([[[[1., 1., 1., 1., 1.],
                     [1., 1., 0., 0., 0.],
                     [1., 1., 0., 0., 0.],
                     [1., 1., 0., 0., 0.],
                     [1., 1., 0., 0., 0.]],
                    [[1., 1., 1., 1., 1.],
                     [1., 0., 1., 0., 1.],
                     [1., 0., 1., 0., 1.],
                     [1., 0., 1., 0., 1.],
                     [1., 0., 1., 0., 1.]],
                    [[1., 0., 0., 1., 0.],
                     [1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1.],
                     [1., 1., 1., 1., 1.],
                     [1., 0., 0., 1., 0.]],
                    [[1., 0., 1., 0., 0.],
                     [1., 0., 1., 0., 0.],
                     [1., 0., 1., 0., 0.],
                     [1., 1., 1., 1., 1.],
                     [1., 0., 1., 0., 0.]]]])
                  
test_arrays = list(zip(test_x, test_y))

@pytest.mark.parametrize("shapes", [[1, 4, 5, 2], [10, 4, 5, 2],
                                    [1, 1, 1, 1], [10, 10, 10, 10]])
def test_generate_labels_returns_correct_shapes(shapes):
    assert np.array_equal(combigen.generate_labels(*shapes).shape, shapes)

@pytest.mark.parametrize("shapes", [[4, 5, 2], [1, 1, 1], [5, 5, 5]])
def test_generate_labels_doesnt_return_all_true(shapes):
    # Generate a large number of y values to test
    large_test_Y = combigen.generate_labels(1000000, *shapes)

    n, stacks, size, dims = large_test_Y.shape

    # Sum over the long dimension of each sample to see how many of them are
    # set to the on state. If they are all on, then it will sum to the length of
    # the dim.
    label_sums = np.sum(large_test_Y, axis=2)
    
    expected_values_in_sum = np.isin(range(size + 1), label_sums)
    assert len(expected_values_in_sum) == len(range(size + 1))
    
    # Get unique values in the sum and their counts and put them in a dict
    count_dict = {val:count for val, count in zip(
        *np.unique(label_sums, return_counts=True))}
    # Sanity check
    assert sum(count_dict.values()) == n * dims * stacks
    
    # Perform the actual check
    assert all(expected_values_in_sum == size*[True] + [False])        

def test_inverse_transform():
    gen_X = combigen.inverse_transform(np.array(test_y))
    assert np.array_equal(test_x, gen_X)        
