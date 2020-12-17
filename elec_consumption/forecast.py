"""Functions for forecast."""
import logging
import sys
from typing import Callable

from fbprophet import Prophet
from loguru import logger
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from statsmodels.tsa.arima.model import ARIMA

logging.disable(sys.maxsize)


def fit_sarima(df: Series) -> Series:
    """Fit a SARIMA model with 1 AR and 1 7-season AR.

    Args:
        df: training data.

    Returns:
        Three forecasted values.
    """
    mod = ARIMA(
        endog=df, order=(1, 0, 0), freq='D',
        seasonal_order=(1, 0, 0, 7),
    ).fit()
    return mod.forecast(3)


def fit_prophet(df: Series) -> Series:
    ts = df.to_frame().reset_index()  # to Prophet time series
    ts.columns = ['ds', 'y']

    mod = Prophet()
    mod.fit(ts)

    future = mod.make_future_dataframe(periods=3)
    forecast = mod.predict(future)

    res = forecast[['ds', 'yhat']].tail(3)
    res.set_index('ds', inplace=True)
    res = res['yhat']
    res.index.name = None

    return res


def validate(
    func: Callable, df: DataFrame, last_training_idx: int
) -> DataFrame:
    """Validate 

    Args:
        last_training_idx: index of the last training date.

    Returns:
        Dataframe with two columns, residuals and last training date.
    """
    if df.shape[0] != 122:
        logger.error('Length of passed dataframe is not 122.')
        res = None
    else:
        res = (
            func(df[:(last_training_idx + 1)]) -
            df[(last_training_idx + 1):(last_training_idx + 4)]
        )
        res = res.to_frame()
        res['last_training_date'] = df.index[last_training_idx]
        res.reset_index(inplace=True)
        res.columns = ['date', 'resid', 'last_training_date']
    return res
