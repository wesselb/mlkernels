import lab as B
from algebra.util import identical
from matrix import Dense
from plum import Dispatcher, Self

from . import _dispatch
from .. import Kernel

__all__ = ["RQ"]


class RQ(Kernel):
    """Rational quadratic kernel.

    Args:
        alpha (scalar): Shape of the prior over length scales. Determines the
            weight of the tails of the kernel. Must be positive.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, alpha):
        self.alpha = alpha

    def _compute(self, dists2):
        return (1 + 0.5 * dists2 / self.alpha) ** (-self.alpha)

    def render(self, formatter):
        return f"RQ({formatter(self.alpha)})"

    @property
    def _stationary(self):
        return True

    @_dispatch(Self)
    def __eq__(self, other):
        return identical(self.alpha, other.alpha)


@_dispatch(RQ, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    return Dense(k._comupte(B.pw_dists2(x, y)))


@_dispatch(RQ, B.Numeric, B.Numeric)
def elwise(k, x, y):
    return k._compute(B.ew_dists2(x, y))
