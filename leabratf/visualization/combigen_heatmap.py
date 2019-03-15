"""Visualization code for the combigen task."""
import logging
from itertools import count

import matplotlib.pyplot as plt
import seaborn as sns

import leabratf.tasks.combinatorics.combigen as cg

logger = logging.getLogger(__name__)

def heatmap(data, data2=None, vmin=0, vmax=2, cbar=False, linewidths=1,
            square=True, gridspec_kw=None, sharey=True, sharex=True,
            titles=None, squeeze=False, y_label='Samples', x_label=None,
            *args, **kwargs):
    """Wrapper function of ``sns.heatmap`` with some different defaults.
    
    Only changed values are shown, see the documentation for ``sns.heatmap``
    for all available parameters.
        
    Parameters
    ----------
    data : np.ndarray
    	The data to send to the heatmap

    data2 : np.ndarray, optional
    	A second dataset to send to the heatmap in a zip
    
    vmin : float, optional
        Min color. Now set to 0
        
    vmax : float, optional
        Max color. Now set to 2
        
    cbar : bool, optional
        Color bar present or not. Now set to False
        
    linewidths : float, optional
        Width of lines between boxes. Now set to 1
        
    square : bool, optional
        Maintain aspect ratio. Now set to True

    gridspec_kw : dict, optional
    	Dictionary of grid specifications for the subplots

    sharey : bool, optional
    	Have the subplots share the y axis.
    
    sharex : bool, optional
    	Have the subplots share the x axis.

    titles : iterable, optional
    	Titles to add to each of the subplots

    squeeze : bool, optional
    	Make the plots compact.

    y_label : str, optional
    	A label to add to the y axis of the plot

    x_label : str, optional
    	A label to add to the x axis (title) of the plot
    """
    n_samples, stack, _, _ = data.shape
    # Check if another dataset was passed
    hor = stack if data2 is None else stack + data2.shape[1]
    # Place everything in subplots
    fig, axn = plt.subplots(n_samples, hor, sharey=sharey, sharex=sharex,
                            squeeze=squeeze, gridspec_kw=gridspec_kw)
    if x_label is not None:
        fig.suptitle(x_label)

    # Create a titles generator
    if titles is not None:
        titles = iter(titles)
    else:
        titles = count(start=0, step=1)
    # Collect the returned heatmap parameters
    heatmaps = []

    # Loop through and generate the plots
    gen_data = iter(data) if data2 is None else zip(data, data2)
    
    for i, sample in enumerate(gen_data):
        if data2 is None:
            stacks = [stack for stack in sample]
        else:
            stacks = [stack for data in sample for stack in data]
            
        for j, stack in enumerate(stacks):
            if len(stack.shape) == 3 and stack.shape[0] == 1:
                stack = stack.reshape(stack.shape[1:])
        
            heatmaps.append(sns.heatmap(
                stack, vmin=vmin, vmax=vmax, cbar=cbar, xticklabels=False,
                yticklabels=False,
                linewidths=linewidths, square=square, ax=axn[i,j], *args,
                **kwargs))
            axn[i,j].set_title(next(titles))
            
            if j is 0:
                axn[i,j].set_ylabel(i, rotation=0)
    
    # Common Labels
    if y_label is not None:
        fig.text(0.05, 0.5, y_label, va='center', rotation='vertical')
    
    return heatmaps

def visualize_combigen(n_pairs=1, *args, **kwargs):
    """Plot ``n_pairs`` of ``X`` and ``y`` from the combigen task

    Parameters
    ----------
    n_pairs : int, optional
    	The number of ``X``, ``y`` pairs to visualize
    """
    heatmaps = []
    # Generate a signle y
    example_y = cg.generate_labels(n_samples=n_pairs, *args, **kwargs)
    # Generate a single x from the y
    example_x = cg.inverse_transform(example_y)

    _, stack, size, dim = example_y.shape
    # Hack
    titles = [a+str(i) for a in ['y','x'] for i in range(4)] + \
             ['']*(n_pairs-1)*stack*2
    gridspec_kw={'width_ratios': [dim]*stack + [size]*stack}

    heatmaps.append(heatmap(example_y, example_x,
                            gridspec_kw=gridspec_kw,
                            sharex=False,
                            titles=titles,
                            x_label='y and X Pairs'))
    return heatmaps
