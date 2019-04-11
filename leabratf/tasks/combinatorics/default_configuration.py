"""File to store the default experimental configuration."""
import logging

logger = logging.getLogger(__name__)

# Combigen Task Variables

# Number of slots in a training set
slots = 4
# Size of each axis in the input array
size = 5
# Number of axes to use per slot
dims = 2
# Number of lines per slot 
lines = 2
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
