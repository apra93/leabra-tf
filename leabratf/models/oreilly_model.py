"""Script to hold the basic O'Reilly Model."""

import logging

import tensorflow as tf

import leabratf.tasks.combinatorics.default_configuration as config
from leabratf.tfutils import lazy_property

logging = logger.getLogger(__name__)

class OReillyModel:
    def __init__(self,
                 inputs,
                 name='or_model',
                 n_inputs=config.n_inputs,
                 n_hidden_1=config.n_hidden_1,
                 n_outputs=config.n_outputs):
        self.name = name 
        self.inputs = inputs

        # Architecture
        self.n_inputs = n_inputs
        self.n_hidden_1 = n_hidden_1
        self.n_outputs = n_outputs

        # Weight initializers
        self.weights_initializer = tf.contrib.layers.xavier_initializer
        self.bias_initializer = tf.zeros_initializer


        with tf.variable_scope(self.name):
            self.weights = {'h1': tf.get_variable(
                                    name='w_h1',
                                    shape=[N_INPUTS, N_HIDDEN_1],
                                    initializer=self.initializer()),
                            'out': tf.get_variable(
                                    name='w_out', 
                                    shape=[N_HIDDEN_1, N_OUTPUTS],
                                    initializer=self.initializer()),
                            }
            self.biases = {'b1': tf.get_variable(
                                name="b_1", 
                                shape=[N_HIDDEN_1], 
                                initializer=self.bias_initializer()),
                            'out': tf.get_variable(
                                name="b_out", 
                                shape=[N_OUTPUTS], 
                                initializer=self.bias_initializer()),
                            }
            self.logits
        
    @lazy_property
    def logits(self):
        """Logits lazy property that returns the logits of the model"""
        with tf.variable_scope('prediction'):
            # Reshape for hidden layer
            self.inp_reshaped = tf.reshape(self.inputs, shape=[-1, N_INPUTS])
            # Single hidden layer
            self.h1_layer_logits = tf.sigmoid(tf.add(tf.matmul(
                self.inp_reshaped, self.weights['h1']), self.biases['b1']))
            # Output layer
            self.out_layer_logits = tf.add(tf.matmul(
                self.h1_layer_logits, self.weights['out']), self.biases['out'])
            # Reshape for labels
            return tf.reshape(self.out_layer_logits, shape=[-1, STACK, SIZE, DIMS])
