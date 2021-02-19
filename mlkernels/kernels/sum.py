import lab as B
from algebra import SumFunction

from . import _dispatch
from .. import Kernel

__all__ = ["SumKernel"]


class SumKernel(Kernel, SumFunction):
    """Sum of kernels."""

    @property
    def _stationary(self):
        return self[0].stationary and self[1].stationary


@_dispatch(SumKernel, object, object)
def pairwise(k, x, y):
    return B.add(pairwise(k[0], x, y), pairwise(k[1], x, y))


@_dispatch(SumKernel, object, object)
def elwise(k, x, y):
    return B.add(elwise(k[0], x, y), elwise(k[0], x, y))
