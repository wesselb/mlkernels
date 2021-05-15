import lab as B
from algebra import SumFunction

from . import _dispatch
from ..mean import Mean

__all__ = ["SumMean"]


class SumMean(Mean, SumFunction):
    """Sum of two means."""

    @_dispatch
    def __call__(self, x):
        return B.add(self[0](x), self[1](x))
