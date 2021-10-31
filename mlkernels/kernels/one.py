import lab as B
from algebra import OneFunction
from matrix import Constant

from . import _dispatch
from .. import Kernel
from ..util import num_elements

__all__ = ["OneKernel"]


class OneKernel(Kernel, OneFunction):
    """Constant kernel of `1`."""

    @property
    def _stationary(self):
        return True


@_dispatch
def pairwise(k: OneKernel, x: B.Numeric, y: B.Numeric):
    return Constant(
        B.one(x, y),
        *B.shape_batch_broadcast(x, y),
        num_elements(x),
        num_elements(y),
    )


@_dispatch
def elwise(k: OneKernel, x: B.Numeric, y: B.Numeric):
    return B.ones(B.dtype(x, y), *B.shape_batch_broadcast(x, y), num_elements(x), 1)
