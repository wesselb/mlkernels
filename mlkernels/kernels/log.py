import lab as B
from matrix import Dense
from plum import Dispatcher, Self

from . import _dispatch
from .. import Kernel

__all__ = ["LogKernel"]


class LogKernel(Kernel):
    """Logarithm kernel."""

    _dispatch = Dispatcher(in_class=Self)

    def render(self, formatter):
        return "LogKernel()"

    @property
    def _stationary(self):
        return True

    @_dispatch(Self)
    def __eq__(self, other):
        return True


@_dispatch(LogKernel, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    dists = B.maximum(B.pw_dists(x, y), 1e-10)
    return Dense(B.divide(B.log(dists + 1), dists))


@_dispatch(LogKernel, B.Numeric, B.Numeric)
def elwise(k, x, y):
    dists = B.maximum(B.ew_dists(x, y), 1e-10)
    return B.divide(B.log(dists + 1), dists)
