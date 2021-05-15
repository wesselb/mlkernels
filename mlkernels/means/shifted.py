import lab as B
from algebra import ShiftedFunction

from . import _dispatch
from ..mean import Mean

__all__ = ["ShiftedMean"]


class ShiftedMean(Mean, ShiftedFunction):
    """Shifted mean."""

    @_dispatch
    def __call__(self, x):
        return self[0](B.subtract(x, self.shifts[0]))
