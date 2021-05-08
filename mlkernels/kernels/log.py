import lab as B
from matrix import Dense

from . import _dispatch
from .. import Kernel

__all__ = ["LogKernel"]


class LogKernel(Kernel):
    """Logarithm kernel."""

    def render(self, formatter):
        return "LogKernel()"

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "LogKernel"):
        return True


@_dispatch
def pairwise(k: LogKernel, x: B.Numeric, y: B.Numeric):
    dists = B.maximum(B.pw_dists(x, y), 1e-10)
    return Dense(B.divide(B.log(dists + 1), dists))


@_dispatch
def elwise(k: LogKernel, x: B.Numeric, y: B.Numeric):
    dists = B.maximum(B.ew_dists(x, y), 1e-10)
    return B.divide(B.log(dists + 1), dists)
