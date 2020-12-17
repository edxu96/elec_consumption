"""Utility functions."""
import time
from typing import Tuple

from pandas.core.series import Series


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
