import lab as B
from matrix import Dense
from plum import Dispatcher, Self

from . import _dispatch
from .. import Kernel

__all__ = ["EQ"]


class EQ(Kernel):
    """Exponentiated quadratic kernel."""

    _dispatch = Dispatcher(in_class=Self)

    def _compute(self, dists2):
        return B.exp(-0.5 * dists2)

    @property
    def _stationary(self):
        return True

    @_dispatch(Self)
    def __eq__(self, other):
        return True


@_dispatch(EQ, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    return Dense(k._compute(B.pw_dists2(x, y)))


@_dispatch(EQ, B.Numeric, B.Numeric)
def elwise(k, x, y):
    return k._compute(B.ew_dists2(x, y))
