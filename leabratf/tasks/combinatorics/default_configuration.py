"""File to store the default experimental configuration."""
import logging
from types import ModuleType
from pstar import pdict

logger = logging.getLogger(__name__)

# Combigen Task Variables

# Number of slots in a training set
slots = 4
# Size of each axis in the input array
size = 5
# Number of axes to use per slot
dims = 2
# Number of lines per slot 
n_lines = 2
# Line frequency statistics
line_stats = [[1,1,1,1,1], [1,1,1,1,1]]

# Data Parameters

# Number of epochs to train for
epochs = 500
# Number of samples in the training set
n_train = 100
# Number of samples in the validation set
n_val = 50
# Number of samples in the testing set
n_test = 500
# General variable that is meant to change based on the context
n_samples = n_train

# Network parameters

# Learning rate
lr = 0.01
# Batch size
batch_size = 1
# Number of parameters in the inputs
n_inputs = slots * size ** dims
# Number of hidden units
n_hidden_1 = 100
# Number of parameters in the labels
n_outputs = slots * size * dims

# Training Parameters

# Number of times to print an update
n_updates = 2
# Which device to train on
tf_device = '/cpu:0'
# Number of models to train with
n_models = 10
# Recompute the model accuracy after this many epochs
n_epochs_acc = 25
# Optimizer
optimizer = 'sgd'

# Meta data variables
# Config name
_name = 'default'
# This is a configuration dict
_config = True

def default_config(**kwargs):
    """Function that turns all the values defined above into a dictionary.

    Grabs all the values in the ``globals`` dictionary and adds them to the
    returned dictionary if it is a relevant variable.

    The function also accepts keyword arguments that will be added to the config
    at the end, overwriting the existing value, or adding the key/value pair.

    Returns
    -------
    default_config : pdict
    	Default configuration as defined by in the default_configuration.py file
    """
    # Configuration in a dictionary format
    locals_dict = globals()
    # Empty dict we will fill
    _dict = pdict()

    # Remove all the non configuration 
    for key, val in locals_dict.items():
        # Skip if the key is a dunder key
        if key.startswith('__') and key.endswith('__'):
            continue
        # Skip if the value is a module
        if isinstance(val, ModuleType):
            continue
        # Skip if the value is a callable (presumably a function)
        if callable(val):
            continue
        # Skip if the value is the logger
        if val is logger:
            continue

        # Passed all skips, add to the dict
        _dict[key] = val

    # Add any kwargs to the dictionary, overwriting any conflicting values
    _dict.update(**kwargs)
    return _dict
