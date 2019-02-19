"""
Tests for leabratf.utils.py
"""
import logging
from pathlib import Path
from collections.abc import Iterable

import pytest
import numpy as np

from leabratf import utils

logger = logging.getLogger(__name__)

test_values = [2, np.pi, True, "test_s", "10", ["test"], ("test",), {"test":1}]
test_lists = [[1,2,3,4,5], [[1],[2],[3],[4],[5]], [[1,2,3],[4,5]],
              [[1,[2,[3,[4,[5]]]]]]]

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

@pytest.mark.parametrize("shape", [(1,1), (1,2), (2,1), (100,100)])
def test_make_input_3d_works_on_arrays(shape):
    @utils.make_input_3d
    def test_func(arg):
        return arg
    assert np.array_equal(test_func(np.ones(shape)).shape, (1, *shape))

@pytest.mark.parametrize("inputs", [(1,), (1,1,1), (1,1,1,1)])
def test_make_input_3d_doesnt_work_on_certain_inputs(inputs):
    @utils.make_input_3d
    def test_func(arg):
        return arg
    assert np.array_equal(test_func(np.ones(inputs)).shape, inputs)
    assert test_func(str(inputs)) == str(inputs)
    
