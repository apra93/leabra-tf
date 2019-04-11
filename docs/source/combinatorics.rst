.. mdinclude:: ../../leabratf/tasks/combinatorics/README.md

   
Default Experimental Configuration
----------------------------------

Below is the default experimental conditions used for the task. Variables
pertaining to the task itself are likely pulled straight from the paper.

.. literalinclude:: ../../leabratf/tasks/combinatorics/default_configuration.py
   :language: python

      
Combigen Task Functions
-----------------------

These are the lower level task generation functions used to put together the
arrays used for inputs and labels.

.. autofunction:: leabratf.tasks.combinatorics.combigen.generate_labels

.. autofunction:: leabratf.tasks.combinatorics.combigen.inverse_transform
   
.. autofunction:: leabratf.tasks.combinatorics.combigen.inverse_transform_single_sample


Dataset Generation
------------------

Below are the functions used to actually generate the datasets used by the
models.

.. autofunction:: leabratf.tasks.combinatorics.make_datasets.generate_combigen_x_y_dataset

.. autofunction:: leabratf.tasks.combinatorics.make_datasets.generate_combigen_datasets

.. autofunction:: leabratf.tasks.combinatorics.make_datasets.generate_iters_inits_and_handle

.. autofunction:: leabratf.tasks.combinatorics.make_datasets.generate_combigen_tf_datasets   
      
