"""Script for tensorflow utility functions and classes"""

import logging
from functools import wraps

import tensorflow as tf

from leabratf import utils

logger = logging.getLogger(__name__)


@utils.doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """The operations added by the function live within a
    ``tf.variable_scope()``.

    If this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped function.
    """
    name = scope or function.__name__
    @wraps(function)
    def decorator(self):
        with tf.variable_scope(name):
            return function(self, *args, **kwargs)
    return decorator

def lazy_property(function):
    """The wrapped method will only be executed once, and the result will be
    stored in a cache variable.

    Subsequent calls to it will directly return the result so that operations
    are added to the graph only once. This is obviously meant to be used with
    classes.
    """
    attribute = '_cache_' + function.__name__
    @property
    @wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

def repeat(tensor, repeats):
    """Implements ``np.repeat`` but for tensors since this is still not
    implemented in tensorflow.

    This function and its test was taken from this github issue page:
    https://github.com/tensorflow/tensorflow/issues/8246

    Parameters
    ----------
    input : tf.Tensor
    	The tensor to be repeated. 1-D or higher.
    
    repeats : list
    	Number of repeat for each dimension, length must be the same as the
    	number of dimensions in input

    Returns
    -------
    tf.Tensor
    	Has the same type as input. Has the shape of ``tensor.shape * repeats``
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor
