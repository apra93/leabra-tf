"""Script to hold the functions for making combigen datasets."""
import logging

import tensorflow as tf

import leabratf.tasks.combinatorics.default_configuration as config
import leabratf.tasks.combinatorics.combigen as cg

logger = logging.getLogger(__name__)

def generate_combigen_x_y_dataset(n_samples=config.n_train, *args, **kwargs):
    """Function to generate X and y pairs for the combigen task.

    Convenience function that runs ``cg.generate_labels`` followed by
    ``cg.inverse_transform`` to creat the ``X`` and ``y`` pairs for the data.
    The only change is that now the default number of samples is defined as
    ``n_train`` from ``default_configuration.py``.

    See documentation for ``cg.generate_labels`` for all passable arguments.

    Parameters
    ----------
    n_samples : int, optional
    	Number of samples to produce for X and Y.

    Returns
    -------
    x : np.array
        The ``X`` that would have generated the inputted ``y``.

    y : np.array
    	Labels for the generated ``X`` array.
    """
    y = cg.generate_labels(n_samples=n_samples, *args, **kwargs)
    x = cg.inverse_transform(y)
    return x, y

def generate_combigen_datasets(
        n_samples_train=config.n_train,
        n_lines_train=config.n_lines,
        line_stats_train=config.line_stats,
        n_samples_val=config.n_val,
        n_lines_val=config.n_lines,
        line_stats_val=config.line_stats,
        n_samples_test=config.n_test,
        n_lines_test=config.n_lines,
        line_stats_test=config.line_stats,
        *args,
        **kwargs,
        ):
    """Generate the training, validation, and testing set data.

    Higher level wrapper that organizes the dataset into the expected groups of
    training, validation, and testing sets. Default values are defined in
    ``default_configuration.py``.

    Parameters that should not be varied between the three datasets (slots,
    size, etc.) are included as star args and kwargs, while the rest that can
    be varied are given their own variables.

    Parameters
    ----------
    n_samples_train : int, optional
    	Number of samples to return in the training data.

    n_lines_train : int, optional
    	Total number of lines to have per sample in the training data.

    line_stats_train : list or None, optional
    	Statistics for sampling from the ``size x dims`` elements for the
    	training data.
    
    n_samples_val : int, optional
    	Number of samples to return in the validation data.

    n_lines_val : int, optional
    	Total number of lines to have per sample in the validation data.

    line_stats_val : list or None, optional
    	Statistics for sampling from the ``size x dims`` elements for the
    	validation data.
    
    n_samples_test : int, optional
    	Number of samples to return in the testing data.

    n_lines_test : int, optional
    	Total number of lines to have per sample in the testing data.

    line_stats_test : list or None, optional
    	Statistics for sampling from the ``size x dims`` elements for the
    	testing data.

    Returns
    -------
    train_data : tuple, (X, y)
    	Training data with ``X`` and ``y``.

    val_data : tuple, (X, y)
    	Validation data with ``X`` and ``y``.

    test_data : tuple, (X, y)
    	Testing data with ``X`` and ``y``.
    """
    # Training data
    x_train, y_train = generate_combigen_x_y_dataset(
        n_samples=n_samples_train,
        n_lines=n_lines_train,
        line_stats=line_stats_train,
        *args,
        **kwargs)

    # Validation data
    x_val, y_val = generate_combigen_x_y_dataset(
        n_samples=n_samples_val,
        n_lines=n_lines_val,
        line_stats=line_stats_val,
        *args,
        **kwargs)

    # Testing data
    x_test, y_test = generate_combigen_x_y_dataset(
        n_samples=n_samples_test,
        n_lines=n_lines_test,
        line_stats=line_stats_test,
        *args,
        **kwargs)

    # Return as a set of tuples that can also be unpacked into components
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def generate_iters_inits_and_handle(tf_datasets, init_ops=None):
    """Generate the iterators, initializers, and string handler for a list of
    ``tf.Dataset``s.

    In order to allow for switching between different ``tf.Dataset``, create a
    string handler tensor that will swap out what values are in the ``x`` and
    ``y`` tensors.

    To change between datasets, get the string handle returned by the
    ``string_handle()`` method of each iterator, and then pass it into the
    placeholder tensor, ``handle``, through the feed dict.

    Parameters
    ----------
    tf_datasets : list
    	List of ``tf.Datasets`` to get the relevant iterators and initializers
    	for.

    init_ops : list or None, optional
    	List of initial operations to perform. Will have initializer operations
    	added to it.

    Returns
    -------
    x : tf.Tensor
    	Tensor that will contain each of the samples. Actual values will depend
    	on the value of the string handler.
    
    y : tf.Tensor
    	Tensor that will contain each of the labels. Actual values will depend
    	on the value of the string handler.
    
    iterators : list
    	List of dataset iterators that will be used to get the string handles.

    handle : tf.Tensor
    	String handler that will determine which dataset to load data from.

    init_ops : list
    	List of operations to run at the start of a ``tf.Session`` with
    	initializers for each iterator.
    """
    
    # Create the iterators for each of the datasets
    iterators = [data.make_initializable_iterator() for data in tf_datasets]

    # Add the iterator initializers to the initialization operations
    init_ops = init_ops or []
    init_ops = [iterator.initializer for iterator in iterators]

    # Rather than creating separate next elements for the model, the `tf.data`
    # API has a string handler iterator so we can contextually switch the active
    # `Dataset` object, resulting in different values being used for `x` and
    # `y`.
    
    # The way this is done is by defining a `tf.placeholder` variable, which is
    # used first to create a string handler iterator, and later to hold the
    # dataset-indicating string handle. The string handler iterator is what then
    # changes the values of `x` and `y`, naturally also supplying them using the
    # `get_next` method.    

    # The placeholder variable of type string
    handle = tf.placeholder(tf.string, shape=[])
    # Iterator from string handle
    handle_iterator = tf.data.Iterator.from_string_handle(
        handle,
        tf_datasets[0].output_types,
        tf_datasets[0].output_shapes)

    # x and y that will be used in the graph
    x, y = handle_iterator.get_next()

    return x, y, iterators, handle, init_ops
        
def generate_combigen_tf_datasets(
        batch_size_train=config.batch_size,
        batch_size_val=config.n_val,
        batch_size_test=config.n_test,
        init_ops=None,
        *args,
        **kwargs):
    """Generates the combigen ``tf.Dataset``s necessary to run with tensorflow.

    Creates all the tensorflow nodes to properly set up the data pipeline, and
    returns the necessary variables. First generate the numpy arrays of the
    data, then convert them to ``tf.Dataset``s with the specified batch size.

    Then get the iterators, initializers, and string handler, so it can be
    properly iterated through.

    Any extra arguments passed will be sent to ``generate_combigen_datasets``,
    so see its documentation for more detail.

    Parameters
    ----------
    batch_size_train : int, optional
    	Batch size to use for the training datasets.

    batch_size_val : int, optional
    	Batch size to use for the validation datasets.

    batch_size_test : int, optional
    	Batch size to use for the testing datasets.

    init_ops : list or None, optional
    	List of operations to run at the start of a ``tf.Session``.

    Returns
    -------
    combigen_tf_parameters : tuple, (x, y, iterators, handle, init_ops)
    	Tuple of the relevant components for the combigen task. See
    	``generate_iters_inits_and_handle`` for more detail.
    """
    # Generate the numpy array datasets. This is a tuple of three tuples, each
    # containing the training, validation, and testing X and y pairs.
    np_datasets = generate_combigen_datasets(*args, **kwargs)

    # Compile the batch sizes into a list in the correct order (train,val,test)
    batch_sizes = [batch_size_train, batch_size_val, batch_size_test]
    # Turn the numpy data into tensorflow datasets
    tf_data = [tf.data.Dataset.from_tensor_slices(data).repeat().batch(batch)
               for data, batch in zip(np_datasets, batch_sizes)]

    # Create the remaining tf objects and return them
    return generate_iters_inits_and_handle(tf_data, init_ops=init_ops)
