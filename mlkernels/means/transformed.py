from algebra import InputTransformedFunction

from . import _dispatch
from ..mean import Mean
from ..util import uprank

__all__ = ["InputTransformedMean"]


class InputTransformedMean(Mean, InputTransformedFunction):
    """Input-transformed mean."""

    @_dispatch
    def __call__(self, x):
        return self[0](uprank(self.fs[0](uprank(x))))
