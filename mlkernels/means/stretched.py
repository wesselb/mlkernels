import lab as B
from algebra import StretchedFunction

from . import _dispatch
from ..mean import Mean

__all__ = ["StretchedMean"]


class StretchedMean(Mean, StretchedFunction):
    """Stretched mean."""

    @_dispatch
    def __call__(self, x):
        return self[0](B.divide(x, self.stretches[0]))
