import lab as B
from algebra import WrappedFunction
from algebra.util import to_tensor, identical
from plum import Dispatcher, Self

from . import _dispatch
from .zero import ZeroKernel
from .. import Kernel
from ..util import uprank

__all__ = ["PeriodicKernel"]


class PeriodicKernel(Kernel, WrappedFunction):
    """Periodic kernel.

    Args:
        k (:class:`.kernel.Kernel`): Kernel to make periodic.
        scale (tensor): Period.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, k, period):
        WrappedFunction.__init__(self, k)
        self.period = to_tensor(period)

    def _compute(self, x, y):
        @uprank
        def feature_map(z):
            z = B.divide(B.multiply(B.multiply(z, 2), B.pi), self.period)
            return B.concat(B.sin(z), B.cos(z), axis=1)

        return feature_map(x), feature_map(y)

    @property
    def _stationary(self):
        return self[0].stationary

    def render_wrap(self, e, formatter):
        return f"{e} per {formatter(self.period)}"

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and identical(self.period, other.period)


@_dispatch(PeriodicKernel, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    return pairwise(k[0], *k._compute(x, y))


@_dispatch(PeriodicKernel, B.Numeric, B.Numeric)
def elwise(k, x, y):
    return elwise(k[0], *k._compute(x, y))


# Periodicise kernels.


@_dispatch(Kernel, object)
def periodicise(a, b):
    return PeriodicKernel(a, b)


@_dispatch(ZeroKernel, object)
def periodicise(a, b):
    return a
