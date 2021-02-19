import lab as B
from algebra import ScaledFunction

from . import _dispatch
from .. import Kernel

__all__ = ["ScaledKernel"]


class ScaledKernel(Kernel, ScaledFunction):
    """Scaled kernel."""

    @property
    def _stationary(self):
        return self[0].stationary


@_dispatch(ScaledKernel, object, object)
def pairwise(k, x, y):
    return B.multiply(k.scale, pairwise(k[0], x, y))


@_dispatch(ScaledKernel, object, object)
def elwise(k, x, y):
    return B.multiply(k.scale, elwise(k[0], x, y))
