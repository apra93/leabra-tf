"""Tests for the combigen task."""
import logging

import pytest
import numpy as np

from leabratf.tasks.combinatorics import combigen

logger = logging.getLogger(__name__)

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

@pytest.mark.parametrize("shapes", [[1, 4, 5, 2],
                                    [10, 4, 5, 2],
                                    [1, 1, 2, 2],
                                    [10, 10, 10, 10]])
def test_generate_labels_returns_correct_shapes(shapes):
    assert np.array_equal(
        combigen.generate_labels(*shapes).shape, shapes)

def test_inverse_transform():
    gen_X = combigen.inverse_transform(np.array(test_y))
    assert np.array_equal(test_x, gen_X)

@pytest.mark.parametrize("n_lines", range(10))
def test_labels_have_the_correct_number_of_lines(n_lines):
    n_samples, slots, dims = 100, 4, 2
    labels = combigen.generate_labels(
        n_samples=n_samples,
        slots=slots,
        dims=dims,
        n_lines=n_lines)
    # Ensure the sums are exactly what we expect them to be at each combination
    assert labels.sum() == n_lines*slots*n_samples

@pytest.mark.parametrize("idx", range(10))    
def test_label_statistics_correspond_to_correct_element_indices(idx):
    line_stats = np.zeros(10)
    line_stats[idx] = 1
    label = combigen.generate_labels(slots=1, n_lines=1, line_stats=line_stats)
    for i in range(5):
        for j in range(2):
            if i == idx%5 and j==idx//5:
                assert label[0, 0, i, j] == 1
            else:
                assert label[0, 0, i, j] == 0

def test_default_labels_uniformly_samples_lines():
    slots = 100
    # Create an array containing all the sums of labels with `dims` dimensions. 
    y_tests = np.array(
        [combigen.generate_labels(
            n_samples=1, slots=slots, dims=1, n_lines=1).sum(axis=1)
         for _ in range(100)])
    # Assert the standard deviation of the means of the sums is less than 1
    # percent of the total number of slots used.
    assert slots * 0.01 > y_tests.mean(axis=0).std()

def test_nonuniformly_sampled_labels_return_the_correct_statistics():
    slots = 100
    line_stats = np.array([1,2,3,4,5])

    # Create an array containing all the sums of labels with `dims` dimensions. 
    y_tests = np.array(
        [combigen.generate_labels(
            n_samples=1, 
            slots=slots, 
            dims=1, 
            n_lines=1, 
            line_stats=line_stats,).sum(axis=1)
         for _ in range(100)])

    # Separate out the means of sums
    means = y_tests.mean(axis=0).reshape((5))

    # Assert the different pair-wise sums are as expected of the stats defined
    # above.
    for l, m in zip(line_stats, means):
        assert np.isclose(means[0]*l, m, atol=slots*0.05)

