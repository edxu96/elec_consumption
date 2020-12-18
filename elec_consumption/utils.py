"""Utility functions."""
import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from pandas.core.series import Series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class ContextTimer():
    """A context manager to log running time."""

    def __enter__(self):
        """Record the starting time when entering.
        Returns:
            ContextTimer: the manager itself to be used in the context.
        """
        self.start = time.time()
        return self

    def __exit__(self, typ, value, traceback):
        """Log the running time when exiting.
        # noqa: DAR101
        """
        self.duration = time.time() - self.start
        print(f'It took {self.duration} seconds to run.')


def cal_increment_percent(series: Series) -> Tuple[float, float, float]:
    """Calculate percentage of increment for a given series.

    Args:
        series: a Pandas series.

    Returns:
        Increment in percentage.
    """
    _max = series.max()
    _min = series.min()
    _percent = abs((_max - _min) / _min) * 100
    return _percent, _max, _min


def plot_acf_pacf(series: Series, whe_ylim: Optional[bool] = True):
    _, (ax1, ax2) = plt.subplots(2, 1)
    plot_acf(series, lags=60, ax=ax1);
    ax1.set_xlim([0.5, 60.5]);
    if whe_ylim:
        ax1.set_ylim([-0.3, 0.3]);

    plot_pacf(series, lags=60, ax=ax2);
    ax2.set_xlim([0.5, 60.5]);
    if whe_ylim:
        ax2.set_ylim([-0.3, 0.3]);
