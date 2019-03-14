"""
Tests for leabratf.utils.py
"""
import logging
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pytest

from leabratf import utils

logger = logging.getLogger(__name__)

test_values = [2, np.pi, True, "test_s", "10", ["test"], ("test",), {"test":1}]
test_lists = [[1,2,3,4,5],
              [[1],[2],[3],[4],[5]],
              [[1,2,3],[4,5]],
              [[1,[2,[3,[4,[5]]]]]],
]

def test_RotatingFileHandlerRelativePath_instantiates():
    assert utils.RotatingFileHandlerRelativePath('test')

def test_setup_logging_doesnt_raise_an_exception():
    utils.setup_logging()

@pytest.mark.parametrize("test", test_values)
def test_isiterable_correctly_returns(test):
    iterable = utils.isiterable(test)
    if isinstance(test, str):
        assert iterable is False
    elif isinstance(test, Iterable):
        assert iterable is True
    else:
        assert iterable is False

@pytest.mark.parametrize("test", test_lists)
def test_flatten_works_correctly(test):
    assert utils.flatten(test) == [1,2,3,4,5]

@pytest.mark.parametrize("size", [[5,5], [16,9], [20,10]])
def test_set_plot_size_correctly_changes_plot_size(size):
    @utils.set_plot_size(size)
    def inner():
        pass
    gcf = plt.gcf()
    assert all(gcf.get_size_inches() != size)
    inner()
    assert all(gcf.get_size_inches() == size)
