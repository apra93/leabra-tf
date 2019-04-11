"""Tests for the combigen dataset generation."""
import logging

import numpy as np
import tensorflow as tf

import leabratf.tasks.combinatorics.default_configuration as config
from leabratf.tasks.combinatorics.make_datasets import generate_combigen_tf_datasets

def test_combigen_make_tf_datasets_correctly_creates_datasets():
    x, y, iterators, handle, init_ops = generate_combigen_tf_datasets()

    with tf.Session() as sess:
        # Run the initialization
        sess.run(init_ops)
        # Define training and validation handlers
        train_handle, val_handle, test_handle = sess.run(
            [i.string_handle() for i in iterators])
        
        # Get training values
        ret_x_train, ret_y_train = sess.run([x, y],
                                            feed_dict={handle: train_handle})
        # Grab their shapes
        shape_x_train, shape_y_train = ret_x_train.shape, ret_y_train.shape

        # Collect the expected shapes
        expected_x_shape = [config.batch_size,
                            config.slots,
                            config.size,
                            config.size]
        expected_y_shape = [config.batch_size,
                            config.slots,
                            config.size,
                            config.dims]

        # Assert the shapes match
        assert shape_x_train[0] == shape_y_train[0]
        assert np.array_equal(shape_x_train, expected_x_shape)
        assert np.array_equal(shape_y_train, expected_y_shape)

        # Get validation values
        ret_x_val = sess.run(x, feed_dict={handle: val_handle})
        # Get testing values
        ret_x_test = sess.run(x, feed_dict={handle: test_handle})

        # Assert the batch sizes are what they are supposed to be
        assert ret_x_val.shape[0] == config.n_val
        assert ret_x_test.shape[0] == config.n_test
