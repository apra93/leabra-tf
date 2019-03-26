"""
Tests for leabratf.utils.py
"""
import logging

import numpy as np
import tensorflow as tf

from leabratf import tfutils

def test_repeat():
    """
    Test taken from github issue page:
    https://github.com/tensorflow/tensorflow/issues/8246
    """
    def np_repeat(tensor, repeats):
        assert len(repeats) == tensor.ndim, "dimension must match"
        repeated = tensor
        for axis, repeat in enumerate(repeats):
            repeated = np.repeat(repeated, repeat, axis = axis)
        return repeated
    shape = [1,3,3,3,2]
    repeat = [1,2,2,3,1]
    tensor = np.random.randn(*shape)
    np_repeated_tensor = np_repeat(tensor, repeat)
    tf_tensor = tf.constant(tensor)
    g = tf.get_default_graph()
    tf_new = tfutils.repeat(tf_tensor, repeat)
    with tf.Session(graph=g) as sess:
        tf_repeated_tensor = tf_new.eval()
    assert np.allclose(np_repeated_tensor, tf_repeated_tensor)
