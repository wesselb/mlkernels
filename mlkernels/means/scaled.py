import lab as B
from algebra import ScaledFunction

from . import _dispatch
from ..mean import Mean

__all__ = ["ScaledMean"]


class ScaledMean(Mean, ScaledFunction):
    """Scaled mean."""

    @_dispatch
    def __call__(self, x):
        return B.multiply(self.scale, self[0](x))
