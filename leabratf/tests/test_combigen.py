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

@pytest.mark.parametrize("shapes", [[1, 4, 5, 2], [10, 4, 5, 2],
                                    [1, 1, 1, 1], [10, 10, 10, 10]])
def test_generate_labels_returns_correct_shapes(shapes):
    assert np.array_equal(
        combigen.generate_labels(*shapes, [1]*shapes[-1]).shape, shapes)

def test_inverse_transform():
    gen_X = combigen.inverse_transform(np.array(test_y))
    assert np.array_equal(test_x, gen_X)

@pytest.mark.parametrize("line_1", range(5))
@pytest.mark.parametrize("line_2", range(5))
def test_labels_have_the_correct_number_of_lines(line_1, line_2):
    n_samples, stack, dims = 1000, 4, 2
    labels = combigen.generate_labels(n_samples=n_samples, stack=stack,
                                      dims=dims, n_lines=[line_1, line_2])
    # Ensure the sums are exactly what we expect them to be at each combination
    assert np.array_equal(labels.sum(axis=(0,1,2)),
                          [line_1*stack*n_samples, line_2*stack*n_samples])

