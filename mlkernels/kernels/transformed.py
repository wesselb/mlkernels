from algebra import InputTransformedFunction
from algebra.util import identical

from . import _dispatch
from .. import Kernel
from ..util import expand, uprank

__all__ = ["InputTransformedKernel"]


class InputTransformedKernel(Kernel, InputTransformedFunction):
    """Input-transformed kernel."""

    def _compute(self, x, y):
        f1, f2 = expand(self.fs)
        x = x if f1 is None else f1(uprank(x))
        y = y if f2 is None else f2(uprank(y))
        return x, y

    @_dispatch
    def __eq__(self, other: "InputTransformedKernel"):
        return self[0] == other[0] and identical(expand(self.fs), expand(other.fs))


@_dispatch
def pairwise(k: InputTransformedKernel, x, y):
    return pairwise(k[0], *k._compute(x, y))


@_dispatch
def elwise(k: InputTransformedKernel, x, y):
    return elwise(k[0], *k._compute(x, y))
