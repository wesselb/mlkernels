import lab as B
from algebra import ReversedFunction

from . import _dispatch
from .. import Kernel

__all__ = ["ReversedKernel"]


class ReversedKernel(Kernel, ReversedFunction):
    """Reversed kernel.

    Evaluates with its arguments reversed.
    """

    @property
    def _stationary(self):
        return self[0].stationary


@_dispatch(ReversedKernel, object, object)
def pairwise(k, x, y):
    return B.transpose(pairwise(k[0], y, x))


@_dispatch(ReversedKernel, object, object)
def elwise(k, x, y):
    return elwise(k[0], y, x)
