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


@_dispatch
def pairwise(k: ZeroKernel, x: B.Numeric, y: B.Numeric):
    return Zero(
        B.dtype(x, y),
        *B.shape_batch_broadcast(x, y),
        num_elements(x),
        num_elements(y),
    )


@_dispatch
def elwise(k: ZeroKernel, x: B.Numeric, y: B.Numeric):
    return B.zeros(
        B.dtype(x, y),
        *B.shape_batch_broadcast(x, y),
        num_elements(x),
        1,
    )
