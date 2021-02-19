import lab as B
from matrix import Dense
from plum import Dispatcher, Self

from . import _dispatch
from .. import Kernel

__all__ = ["Matern52"]


class Matern52(Kernel):
    """Matern--5/2 kernel."""

    _dispatch = Dispatcher(in_class=Self)

    def _compute(self, dists):
        r1 = 5 ** 0.5 * dists
        r2 = 5 * dists ** 2 / 3
        return (1 + r1 + r2) * B.exp(-r1)

    @property
    def _stationary(self):
        return True

    @_dispatch(Self)
    def __eq__(self, other):
        return True


@_dispatch(Matern52, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    return Dense(k._compute(B.pw_dists(x, y)))


@_dispatch(Matern52, B.Numeric, B.Numeric)
def elwise(k, x, y):
    return k._compute(B.ew_dists(x, y))
