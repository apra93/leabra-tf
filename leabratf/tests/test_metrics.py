"""Tests for the visualization functions for model metrics"""
import logging

import pandas as pd
import pytest

from leabratf.visualization.metrics import plot_df_str_metrics

logger = logging.getLogger(__name__)

# Two columns, and five rows of string lists
data = [[str(list(range(10))) for _ in range(2)] for _ in range(5)]
columns = ['col1', 'col2']
df = pd.DataFrame(data, columns=columns)

def test_plot_df_str_metrics_passes_default_input_combinations():
    assert plot_df_str_metrics(df) is None

@pytest.mark.parametrize("metrics", [[columns[0]], [columns[1]], columns])
@pytest.mark.parametrize("title", ['', 'Test'])
@pytest.mark.parametrize("epochs", [1, 5])
@pytest.mark.parametrize("key_by_model", [True])
@pytest.mark.parametrize("model_average", [True])
@pytest.mark.parametrize("epoch_vline", [None, 5])
def test_plot_df_str_metrics_passes_all_nondefault_input_combinations(
        metrics, title, epochs, key_by_model, model_average, epoch_vline):
    args =  metrics, title, epochs, key_by_model, model_average, epoch_vline
    assert plot_df_str_metrics(df, *args) is None



