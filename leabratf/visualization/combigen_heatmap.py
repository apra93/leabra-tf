"""Visualization code for the combigen task."""
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from leabratf.utils import make_input_3d

logger = logging.getLogger(__name__)

# General warning, this will overwrite the originally defined heatmap
@make_input_3d
def heatmap(data, vmin=0, vmax=2, cbar=False, linewidths=1, square=True, 
            samples_per_row=10, gridspec_kw=None, sharey=True, sharex=True,
            *args, **kwargs):
    """Wrapper function of `sns.heatmap` with some different defaults. 
    
    Only changed values are shown, see the documentation for `sns.heatmap`
    for all available parameters.
        
    Parameters
    ----------
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
        
    samples_per_row : int, optional
        Number of samples to have in a plotting row before creating a 
        new row

    gridspec_kw : dict, optional
    	Dictionary of grid specifications for the subplots

    sharey : bool, optional
    	Have the subplots share the y axis.
    
    sharex : bool, optional
    	Have the subplots share the x axis.
    """
    # Place them all in a subplot
    n_samples = len(data)
    ver_size = n_samples // samples_per_row
    ver_size = ver_size + 1 if n_samples % samples_per_row else ver_size
    
    hor_size = (samples_per_row 
                if (ver_size > 1 or n_samples == samples_per_row) 
                else n_samples % samples_per_row)
    
    _, axn = plt.subplots(ver_size, hor_size, sharey=sharey, sharex=sharex,
                          squeeze=False, gridspec_kw=gridspec_kw)
    
    # Loop through and generate the plots
    gen_data = iter(data)
    for i in range(ver_size):
        try:
            for j in range(hor_size):
                array = next(gen_data)
                shape = array.shape
                if len(shape) == 3 and shape[0] == 1:
                    array = array.reshape(shape[1:])
                sns.heatmap(array, vmin=vmin, vmax=vmax, cbar=cbar,
                            linewidths=linewidths, square=square, ax=axn[i,j], 
                            *args, **kwargs)
                axn[i,j].set_title(i*ver_size + j)
        except StopIteration:
            break
