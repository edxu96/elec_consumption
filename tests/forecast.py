"""Test functions for regression."""
from elec_consumption.forecast import fit_sarima
import pandas as pd
from pandas.core.frame import DataFrame
import pytest

_num_units = 2  # How many profiles to be modelled.


@pytest.mark.usefixtures('profiles')
def test_sarima_all(profiles: DataFrame):
    """Build SARIMA for daily profiles.

    Args:
        profiles: 500 household power consumption profiles in half-hour
            resolution.
    """
    fore = {}
    for unit in range(_num_units):
        series = profiles[unit].resample('1D', closed='left').mean()
        assert series.shape[0] == 122
        fore[unit] = fit_sarima(series)

    res = pd.DataFrame.from_dict(fore, orient='columns')
    res.index.name = 'date'
    assert res.shape == (3, _num_units)
    # res.to_csv('three_step_daily.csv')
