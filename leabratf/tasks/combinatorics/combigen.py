"""Combinatorial Generalization Task"""

import logging

import numpy as np

from leabratf.utils import make_input_3d

logger = logging.getLogger(__file__)

def generate_labels(n_samples=1, size=5, dims=2):
    """Returns an array of labels to construct the data from.

    Parameters
    ----------
    n_samples : int, optional
    	Number of samples to return.

    size : int, optional
    	Size of the nxn matrix to use for the task.

    dims : int, optional
    	Number of dimensions for the task.

    Returns
    -------
    labels : np.ndarray (n_samples x size x dims)
    	The resulting task labels.
    """
    return np.random.choice(2, (n_samples, size, dims), True)

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
    n = len(y)
    # Create a horizontal array and a vertical array according to y
    horizontal, vertical = np.tile(y, n).reshape(n, n, 2).T
    return (horizontal.T + vertical).astype(bool)

@make_input_3d
def inverse_transform(Y, *args, **kwargs):
    """Wrapper for first pass implementation that accounts for multiple
    samples.

    Parameters
    ----------
    Y : array-like (mxnx2)
    	Array of samples

    Returns
    -------
    X : list of np.ndarrays
    	List of samples transformed from Y to X.
    """
    return [inverse_transform_single_sample(y) for y in Y]
