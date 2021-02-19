import lab as B
from algebra import ZeroFunction
from matrix import Zero

from . import _dispatch
from .. import Kernel
from ..util import num_elements

__all__ = ["ZeroKernel"]


class ZeroKernel(Kernel, ZeroFunction):
    """Constant kernel of `0`."""

    @property
    def _stationary(self):
        return True


@_dispatch(ZeroKernel, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    return Zero(B.dtype(x), num_elements(x), num_elements(y))


@_dispatch(ZeroKernel, B.Numeric, B.Numeric)
def elwise(k, x, y):
    return B.zeros(B.dtype(x), num_elements(x), 1)
