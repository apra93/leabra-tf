"""Combinatorial Generalization Task"""

import logging

import numpy as np

from leabratf.utils import make_input_3d

logger = logging.getLogger(__file__)

def generate_labels(n_samples=1, stack=4, size=5, dims=2):
    """Returns an array of labels to construct the data from.

    Parameters
    ----------
    n_samples : int, optional
    	Number of samples to return.

    stack : int, optional
    	Number of labels per stack.

    size : int, optional
    	Size of the nxn matrix to use for the task.

    dims : int, optional
    	Number of dimensions for the task.

    Returns
    -------
    labels : np.ndarray (n_samples x size x dims)
    	The resulting task labels.
    """
    # Generate baseline labels
    raw_labels = np.random.choice(2, (n_samples, stack, size, dims), replace=True)
    # Random selection of indices to zero out
    arg_zero = np.random.choice(size, (n_samples*dims*stack), replace=True)
    # Alternating indices to loop through the dims of the labels
    dim_indices = np.tile(range(dims), stack*n_samples)
    # Repeating indices to loop through the samples
    sample_indices = np.repeat(range(n_samples), dims*stack)
    # Stack indices
    stack_indices = np.repeat(np.tile(range(stack), n_samples), dims)
    
    # Zero out a random selection of indices
    raw_labels[sample_indices, stack_indices, arg_zero, dim_indices] = 0
    return raw_labels    

def inverse_transform_single_sample(y):
    """Turns the inputted nxn array into the nx2 array
    
    Parameters
    ----------
    y : array-like (nx2)
        The the label we are transforming.
        
    Returns
    -------
    x : np.array (nxn)
        The `x` that would have generated the inputted `y`.
    """
    # Grab the length of y
    size, _ = y.shape
    # Create a horizontal array and a vertical array according to y
    horizontal, vertical = np.tile(y, size).reshape(size, size, 2).T
    return (horizontal.T + vertical).astype(bool).astype(np.float32)    

def inverse_transform(Y, *args, **kwargs):
    """Wrapper for first pass implementation that accounts for multiple
    samples.

    Parameters
    ----------
    Y : array-like
    	Array of samples

    Returns
    -------
    X : np.ndarray
    	Array of samples transformed from Y to X.
    """
    n_samples, stack, size, dims = Y.shape
    return np.concatenate([inverse_transform_single_sample(face) 
                           for y in Y for face in y], axis=0).reshape(
                                   n_samples, stack, *([size]*dims))
