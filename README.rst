======================================
Leabra Tensorflow - Under Construction
======================================

An (eventual) tensorflow implementation of the the "Local, Error-driven and
Associative, Biologically Realistic Algorithm" (LEABRA). For details see the
`wikipedia <https://en.wikipedia.org/wiki/Leabra>`_ or
`emergent <https://grey.colorado.edu/emergent/index.php/Leabra>`_ pages.

Implementation is drawn from Randall C. O'Reilly's
`Generalization in Interactive Networks: The Benefits of Inhibitory Competition and Hebbian Learning <https://www.mitpressjournals.org/doi/10.1162/08997660152002834>`_.


Notes
-----

- `leabra7 <https://github.com/cdgreenidge/leabra7>`_, a pytorch implementation


TODO
----

- Implement the task described in O'Reilly's paper
- Establish a baseline on the task using ``leabra7`` and other canonical DL architectures
- Reconstruct leabra in tensorflow
- Ensure similar results between ``leabra7`` and ``leabra-tf``
- Pull out inhibitory competition and hebbian learning as learning rules
- Modularize and productionize for future use

Some Useful Links
-----------------

- `Markdown cheatsheet <https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet>`_
- `reStructuredText cheatsheet <https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst>`_
- `reStructuredText online editor <http://rst.ninjs.org/>`_ 

*Will be removed upon 0.0 release.*

Requirements
------------

To be filled out upon stable 0.0 release

Installation
------------

To be filled out upon stable 0.0 release

Running the Tests
-----------------
::

  $ python run_tests.py
   
Directory Structure
-------------------

This repo is based on two cookiecutter templates. See the following github pages for more info:

- `cookiecutter-data-science-pp <https://github.com/apra93/cookiecutter-data-science-pp>`_
- `cookiecutter-data-science <https://github.com/drivendata/cookiecutter-data-science>`_
 
