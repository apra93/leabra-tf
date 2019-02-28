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
