"""Utility functions."""
import time


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
