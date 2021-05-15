import algebra
from algebra import Function

from . import _dispatch

__all__ = ["Mean"]


class Mean(Function):
    """Mean function.

    Means can be added and multiplied.
    """

    @_dispatch
    def __call__(self, x):
        """Construct the mean for a design matrix.

        Args:
            x (input): Points to construct the mean at.

        Returns:
            column vector: Mean vector as a rank-2 column vector.
        """
        raise RuntimeError(f'For mean {self}, could not resolve argument "{x}".')


# Register the algebra.
@algebra.get_algebra.dispatch
def get_algebra(a: Mean):
    return Mean
