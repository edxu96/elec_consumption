"""Configuration and fixtures for tests."""
import pandas as pd
from pandas.core.frame import DataFrame
import pytest


@pytest.fixture(scope='module')
def profiles() -> DataFrame:
    """Prepare 500 household power consumption profiles.

    Returns:
        500 household power consumption profiles in half-hour
        resolution.
    """
    res = pd.read_csv('./data/raw.csv', index_col=0)
    res.columns = list(range(res.shape[1]))
    res.columns.name = 'household'
    res.index = pd.to_datetime(res.index)
    res.index.name = 'datetime'

    assert res.shape == (5856, 500)
    return res
