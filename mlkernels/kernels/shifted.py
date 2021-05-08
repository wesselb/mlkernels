import lab as B
from algebra import ShiftedFunction, shift
from algebra.util import identical

from . import _dispatch
from .. import Kernel
from ..util import expand

__all__ = ["ShiftedKernel"]


class ShiftedKernel(Kernel, ShiftedFunction):
    """Shifted kernel."""

    def _compute(self, x, y):
        shifts1, shifts2 = expand(self.shifts)
        return B.subtract(x, shifts1), B.subtract(y, shifts2)

    @property
    def _stationary(self):
        if len(self.shifts) == 1:
            return self[0].stationary
        else:
            # NOTE: Can do something more clever here.
            return False

    @_dispatch
    def __eq__(self, other: "ShiftedKernel"):
        identical_shifts = identical(expand(self.shifts), expand(other.shifts))
        return self[0] == other[0] and identical_shifts


@_dispatch
def pairwise(k: ShiftedKernel, x: B.Numeric, y: B.Numeric):
    return pairwise(k[0], *k._compute(x, y))


@_dispatch
def elwise(k: ShiftedKernel, x: B.Numeric, y: B.Numeric):
    return elwise(k[0], *k._compute(x, y))


# Make shifting synergise with stationary kernels.


@shift.dispatch
def shift(a: Kernel, *shifts):
    if a.stationary and len(shifts) == 1:
        return a
    else:
        return ShiftedKernel(a, *shifts)
