import lab as B
from algebra import SelectedFunction
from algebra.util import identical

from . import _dispatch
from .. import Kernel
from ..util import expand, uprank

__all__ = ["SelectedKernel"]


class SelectedKernel(Kernel, SelectedFunction):
    """Kernel with particular input dimensions selected."""

    @property
    def _stationary(self):
        if len(self.dims) == 1:
            return self[0].stationary
        else:
            # NOTE: Can do something more clever here.
            return False

    @_dispatch
    def __eq__(self, other: "SelectedKernel"):
        return self[0] == other[0] and identical(expand(self.dims), expand(other.dims))


@_dispatch
def pairwise(k: SelectedKernel, x: B.Numeric, y: B.Numeric):
    return pairwise(k[0], *_selectedkernel_compute(k, x, y))


@_dispatch
def elwise(k: SelectedKernel, x: B.Numeric, y: B.Numeric):
    return elwise(k[0], *_selectedkernel_compute(k, x, y))


@uprank
def _selectedkernel_compute(k, x, y):
    dims1, dims2 = expand(k.dims)
    x = x if dims1 is None else B.take(x, dims1, axis=-1)
    y = y if dims2 is None else B.take(y, dims2, axis=-1)
    return x, y
