"""Plotting functions for metrics obtained throughtout the repo."""
from ast import literal_eval
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from leabratf.utils import set_plot_size

logger = logging.getLogger(__name__)

@set_plot_size()
def plot_df_metrics(metrics_df, metrics=None, title='Training History',
                    epochs=None, key_by_model=False, model_average=False,
                    epoch_vline=500):
    """Plots a dataframe of metrics for multiple models.

    This function was written to plot multiple metrics of 10 models. The metrics
    were stored as csv files, and then read in using `pd.read_csv`, which would
    load the actual metrics as a long `str`. If the data is used upon
    generation, then it won't be of type `str` and will just be used as-is.

    To plot the data, each of the metrics for each model are run using
    `literal_eval` to turn the data into `list`s if they are of type `str`, and
    then `dict`s, the data is then subselected for the desired number of epochs,
    and then compiled into a long-form list of values. These values are then
    passed into `sns.lineplot` for the actual plots.

    See `nb-0.3.1` for example usage.

    Note: Setting `model_average` to `True` for large sets of data will result
    in very long runtimes. 

    Parameters
    ----------
    metrics_df : pd.DataFrame
    	The dataframe containing the model data for each of the models. Data is
    	expected to be of type `str`, which can be evaluated to data of type
    	`list`.

    metrics : list or None, optional
    	Metrics from `metrics_df` to plot (column names). Defaults to plot all
    	of them.

    title : str, optional
    	Title to plot on the data. Defaults to 'Training History'.
    
    epochs : int or None, optional
    	Number of epochs from the data to plot. Defaults to plot every epoch in
    	the data.

    key_by_model : bool, optional
    	Color the plots by the models that they come from, as opposed to the
    	different metrics. Most useful if comparing model performance on a
    	single metric.

    model_average : bool, optional
    	Plot the average performance across all models on each metric. Warning:
    	Setting this to `True` will result in long runtimes for large datasets.

    epoch_vline : int or None, optional
    	Place to plot a dashed vertical line in the metrics plot. If 
    	`epoch_vline` is `None`, or `epoch_vline > epochs`, then the line will
    	not be plotted.
    """
    # What metrics to plot
    metrics = metrics or metrics_df.columns
    # Empty lists for the long form data
    long_epochs, long_metrics, long_hues, long_units = [], [], [], []
    
    # Loop through each model's data
    for i, metrics_series in metrics_df.iterrows():
        # Series data is in a string format, convert to floats and put them in a
        # dict
        if isinstance(metrics_series[metrics[0]], str):
            metrics_dict = {key: [float(val)
                                  for val in literal_eval(metrics_series[key])]
                            for key in metrics}
        else:
            metrics_dict = {key: [val for val in metrics_series[key]]
                            for key in metrics}
        
        # How many epochs to plot
        if not epochs:
            len_metrics = [len(val) for val in metrics_dict.values()]
            epochs = len_metrics[0]

        # Add to the long form lists
        for key in metrics:
            label = key if not key_by_model else 'Model {0}'.format(i)
            units = i if not key_by_model else key
            long_epochs += list(range(epochs))
            long_metrics += metrics_dict[key][:epochs]
            long_hues += [label]*epochs
            long_units += [units]*epochs

    if model_average:
        sns.lineplot(x=long_epochs, y=long_metrics, hue=long_hues, 
                     estimator='mean')
    # Plot each line individually
    else:
        sns.lineplot(x=long_epochs, y=long_metrics, hue=long_hues, 
                     estimator=None, units=long_units)

    # Title, axis, and 500 epoch line
    plt.title(title)
    plt.xlabel('Epochs')
    if epoch_vline and epochs > epoch_vline:
        plt.axvline(epoch_vline, linestyle='--', label='{0} Epochs'.format(
            epoch_vline))
        
    # Prune down the number of labels to just the unique ones
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
