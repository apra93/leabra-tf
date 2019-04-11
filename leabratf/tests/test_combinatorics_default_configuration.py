"""Tests for the default combinatorics """
import logging

logger = logging.getLogger(__name__)

def test_combinatorial_default_configuration_imports():
    import leabratf.tasks.combinatorics.default_configuration as config

def test_combinatorial_default_configuration_attrs():
    import leabratf.tasks.combinatorics.default_configuration as config
    # Loop through each attribute given by dir and simply call it
    for attr in dir(config):
        # Skip if its a dunder variable
        if '__' not in attr:
            getattr(config, attr)
