import lab as B
from algebra import StretchedFunction
from algebra.util import identical

from . import _dispatch
from .. import Kernel
from ..util import expand

__all__ = ["StretchedKernel"]


class StretchedKernel(Kernel, StretchedFunction):
    """Stretched kernel."""

    def _compute(self, x, y):
        stretches1, stretches2 = expand(self.stretches)
        return B.divide(x, stretches1), B.divide(y, stretches2)

    @property
    def _stationary(self):
        if len(self.stretches) == 1:
            return self[0].stationary
        else:
            # NOTE: Can do something more clever here.
            return False

    @_dispatch
    def __eq__(self, other: "StretchedKernel"):
        identical_stretches = identical(expand(self.stretches), expand(other.stretches))
        return self[0] == other[0] and identical_stretches


@_dispatch
def pairwise(k: StretchedKernel, x: B.Numeric, y: B.Numeric):
    return pairwise(k[0], *k._compute(x, y))


@_dispatch
def elwise(k: StretchedKernel, x: B.Numeric, y: B.Numeric):
    return elwise(k[0], *k._compute(x, y))
