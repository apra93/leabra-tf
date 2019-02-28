"""Combinatorial Generalization Task"""

import logging

import numpy as np

from leabratf.utils import as_list

logger = logging.getLogger(__file__)

def generate_labels(n_samples=1, stack=4, size=5, dims=2, n_lines=[1,1]):
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

    n_lines : list or int, optional
    	Number of lines to have on each axis. If an int is provided, then it is
    	used for each axis.
    
    Returns
    -------
    labels : np.ndarray (n_samples x size x dims)
    	The resulting task labels.

    Raises
    ------
    ValueError
    	If `dims` does not match the number of lines provided (assuming more
    	than one number was provided for it)
    """
    # Ensure this is a list
    n_lines = as_list(n_lines)
    # If one number is passed in for n_lines and there is more than 1 dim, then
    # assume that they should both be set to the value of n_lines.
    if len(n_lines) == 1 and dims != 1:
        n_lines *= dims
    # Ensure dims and `len(n_lines)` is the same
    if dims != len(n_lines):
        raise ValueError('Value for dims must match len(n_lines)')
    
    # Generate a zero array to fill with 1s
    raw_labels = np.zeros((n_samples, stack, size, dims))

    # Create a list of length `dims` that contains arrays with the indices which
    # to set the value to 1. Each array is of shape `n_samples` by `stack` by
    # `n_line[i]` where `i` is the line index.
    arg_ones = [np.array([np.random.choice(range(size), line, replace=False)
                          for _ in range(n_samples*stack)])
                .reshape((n_samples,stack,line))
                for line in n_lines]

    # Use the index arrays created above to set the desired indices of the
    # zero-array to be 1 for each dim in dims.
    for dim, arg_one in enumerate(arg_ones):
        np.put_along_axis(raw_labels[:,:,:,dim], arg_one, values=1, axis=2)
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
