import lab as B
from algebra import ProductFunction

from . import _dispatch
from .. import Kernel

__all__ = ["ProductKernel"]


class ProductKernel(Kernel, ProductFunction):
    """Product of two kernels."""

    @property
    def _stationary(self):
        return self[0].stationary and self[1].stationary


@_dispatch
def pairwise(k: ProductKernel, x, y):
    return B.multiply(pairwise(k[0], x, y), pairwise(k[1], x, y))


@_dispatch
def elwise(k: ProductKernel, x, y):
    return B.multiply(elwise(k[0], x, y), elwise(k[1], x, y))
