"""Combinatorial Generalization Task"""

import logging

import numpy as np

import leabratf.tasks.combinatorics.default_configuration as config
from leabratf.utils import as_list, flatten

logger = logging.getLogger(__file__)


def generate_labels(n_samples=1,
                    slots=config.slots,
                    size=config.size,
                    dims=config.dims,
                    n_lines=config.n_lines,
                    line_stats=config.line_stats):
    """Returns an array of labels to construct the data from.

    Parameters
    ----------
    n_samples : int, optional
    	Number of samples to return.

    slots : int, optional
    	Number of slots per sample.

    size : int, optional
    	Size of the nxn matrix to use for the task.

    dims : int, optional
    	Number of dimensions for the task.

    n_lines : int, optional
    	Total number of lines to have per sample.

    line_stats : list or None, optional
    	Statistics for sampling from the ``size x dims`` elements.
    
    Returns
    -------
    labels : np.ndarray of shape ``(n_samples, stack, size, dims)``
    	The resulting task labels.

    Raises
    ------
    ValueError
    	If ``dims`` does not match the number of lines provided (assuming more
    	than one number was provided for it)    
    """
    # Ensure `n_lines` is an int
    n_lines = int(n_lines)
    # This will be useful going forward
    n_idx = size * dims
    # It must be less than the number of available indices
    if n_lines >= n_idx:
        raise ValueError('n_lines must be less than size * dims.')
        
    # Normalize `line_stats` to sum to 1 if it isn't already
    if line_stats is not None:
        line_stats = flatten(line_stats)
        if sum(line_stats) != 1.0:
            line_stats = np.array(line_stats) / sum(line_stats)
        
    # Generate a zero array to fill with 1s
    raw_labels = np.zeros((n_samples, slots, n_idx))
    
    # Create a list of length `dims` that contains arrays with the indices which
    # to set the value to 1. Each array is of shape `n_samples` by `stack` by
    # `n_line[i]` where `i` is the line index.
    arg_ones = np.array([np.random.choice(range(n_idx), 
                                          n_lines, 
                                          replace=False,
                                          p=line_stats)
                         for _ in range(n_samples * slots)]).reshape(
        (n_samples, slots, n_lines))

    # Use the index arrays created above to set the desired indices of the
    # zero-array to be 1 for each dim in dims.
    np.put_along_axis(raw_labels, arg_ones, values=1, axis=2)
    return raw_labels.reshape((n_samples, slots, size, dims), order='F')

def inverse_transform_single_sample(y):
    """Turns the inputted nxn array into the nx2 array
    
    Parameters
    ----------
    y : array-like (nx2)
        The the label we are transforming.

    Returns
    -------
    X : np.array (nxn)
        The ``X`` that would have generated the inputted ``y``.
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
    	Array of samples transformed from ``Y`` to ``X``.
    """
    n_samples, stack, size, dims = Y.shape
    return np.concatenate([inverse_transform_single_sample(face)
                           for y in Y for face in y], axis=0).reshape(
                                   n_samples, stack, *([size]*dims))
