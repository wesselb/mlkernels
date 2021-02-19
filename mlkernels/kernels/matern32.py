import lab as B
from matrix import Dense
from plum import Dispatcher, Self

from . import _dispatch
from .. import Kernel

__all__ = ["Matern32"]


class Matern32(Kernel):
    """Matern--3/2 kernel."""

    _dispatch = Dispatcher(in_class=Self)

    def _compute(self, dists):
        r = 3 ** 0.5 * dists
        return (1 + r) * B.exp(-r)

    @property
    def _stationary(self):
        return True

    @_dispatch(Self)
    def __eq__(self, other):
        return True


@_dispatch(Matern32, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    return Dense(k._compute(B.pw_dists(x, y)))


@_dispatch(Matern32, B.Numeric, B.Numeric)
def elwise(k, x, y):
    return k._compute(B.ew_dists(x, y))
