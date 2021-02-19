import lab as B
from algebra import StretchedFunction
from algebra.util import identical
from plum import Dispatcher, Self

from . import _dispatch
from .. import Kernel
from ..util import expand

__all__ = ["StretchedFunction"]


class StretchedKernel(Kernel, StretchedFunction):
    """Stretched kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @property
    def _stationary(self):
        if len(self.stretches) == 1:
            return self[0].stationary
        else:
            # NOTE: Can do something more clever here.
            return False

    @_dispatch(Self)
    def __eq__(self, other):
        identical_stretches = identical(expand(self.stretches), expand(other.stretches))
        return self[0] == other[0] and identical_stretches


@_dispatch(StretchedKernel, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    return pairwise(k[0], *_stretchedkernel_compute(k, x, y))


@_dispatch(StretchedKernel, B.Numeric, B.Numeric)
def elwise(k, x, y):
    return elwise(k[0], *_stretchedkernel_compute(k, x, y))


def _stretchedkernel_compute(k, x, y):
    stretches1, stretches2 = expand(k.stretches)
    return B.divide(x, stretches1), B.divide(y, stretches2)
