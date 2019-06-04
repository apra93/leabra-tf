"""Script to hold the basic O'Reilly Model."""
import logging
from collections import OrderedDict

from pstar import pdict
import tensorflow as tf

import leabratf.tasks.combinatorics.default_configuration as default_config
from leabratf.tfutils import lazy_property

logger = logging.getLogger(__name__)



class OReillyModel:
    def __init__(self,
                 inputs,
                 name='or_model',
                 config=None):
        self.name = name 
        self.inputs = inputs
        self.config = self.init_config(config)

        # Layer name and shape dictionary
        self.layers = OrderedDict(
            [('l1', [self.n_inputs, self.n_hidden_1]),
             ('out', [self.n_hidden_1, self.n_outputs]),
            ])
        self.layer_names = self.layers.keys()

        self.logits

        # Saver
        self.saver = tf.train.Saver()
        
    def init_config(self, config=None):
        # Use the default config if none was provided
        config = config or default_config

        # Architecture
        self.n_inputs = config.n_inputs
        self.n_hidden_1 = config.n_hidden_1
        self.n_outputs = config.n_outputs
        self.slots = config.slots
        self.size = config.size
        self.dims = config.dims
        
        # Weight initializers
        self.weight_initializer = config.weights_initializer
        self.bias_initializer = config.bias_initializer

        # Shape combinations
        self.input_shape = [-1, self.n_inputs]
        self.output_shape = [-1, self.slots, self._size, self.dims]

        # Directories
        self.dir_checkpoints = config.dir_checkpoints
        self.dir_tensorboard = config.dir_tensorboard

        return config

    def _get_layer_tf_variables(self, prefix, initializer, layers=None,
                                bias=False):
        layers = layers or self.layers
        return pdict(
            {layer : tf.get_variable(
                name='{0}_{1}'.format(prefix, layer),
                shape=shape if not bias else shape[1:],
                initializer=initializer)
             for layer, shape in layers.items()})

    @lazy_property
    def weights(self):
        return self._get_layer_tf_variables('w', self.weight_initializer)
    
    @lazy_property
    def biases(self):
        return self._get_layer_tf_variables('b', self.bias_initializer,
                                            bias=True)
        
    def logits(self):
        """Logits lazy property that returns the logits of the model"""
        with tf.variable_scope(self.name):
            # Reshape for hidden layer
            self.inp_reshaped = tf.reshape(self.inputs, shape=self.input_shape)
            
            # Single hidden layer
            self.h1_layer_logits = tf.sigmoid(tf.add(
                tf.matmul(self.inp_reshaped, self.weights['h1']),
                self.biases['b1']))
            
            # Output layer
            self.out_layer_logits = tf.add(
                tf.matmul(self.h1_layer_logits, self.weights['out']),
                self.biases['out'])
            
            # Reshape for labels
            return tf.reshape(self.out_layer_logits, shape=self.output_shape)

    def save(self, sess, global_step_tensor, dir_checkpoints=None):
        """Save the checkpoint in the path defined in the config file."""
        self.saver.save(
            sess,
            global_step_tensor,
            dir_checkpoints or self.dir_checkpoints)

    def load(self, sess):
        """Load latest checkpoint from the experiment path."""
        latest_checkpoint = tf.train.latest_checkpoint(
            self.dir_checkpoint)
        if latest_checkpoint:
            self.saver.restore(sess, latest_checkpoint)
