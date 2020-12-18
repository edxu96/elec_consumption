"""Test functions for regression."""
from elec_consumption.forecast import fit_prophet_hourly, fit_sarima
from loguru import logger
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


def test_prophet_all_hourly(profiles: DataFrame):
    """Build Prophet models for hourly power consumption profiles.

    Args:
        profiles: 500 household power consumption profiles in half-hour
            resolution.
    """
    fore = {}
    for unit in range(_num_units):
        # Down-sample to hourly profile.
        series = profiles[unit].resample('1H', closed='left').mean()

        assert series.shape[0] == 122 * 24
        fore[unit] = fit_prophet_hourly(series)
        logger.info(f'4-step forecasts for unit {unit} have been calculated.')

    res = pd.DataFrame.from_dict(fore, orient='columns')
    res.index.name = 'datetime'
    assert res.shape == (4, _num_units)
    # res.to_csv('four_step_hourly.csv')
