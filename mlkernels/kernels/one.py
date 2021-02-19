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


@_dispatch(OneKernel, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    return Constant(B.one(x), num_elements(x), num_elements(y))


@_dispatch(OneKernel, B.Numeric, B.Numeric)
def elwise(k, x, y):
    return B.ones(B.dtype(x), num_elements(x), 1)
