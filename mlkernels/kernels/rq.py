import lab as B
from algebra.util import identical
from matrix import Dense

from . import _dispatch
from .. import Kernel

__all__ = ["RQ"]


class RQ(Kernel):
    """Rational quadratic kernel.

    Args:
        alpha (scalar): Shape of the prior over length scales. Determines the
            weight of the tails of the kernel. Must be positive.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def _compute(self, dists2):
        return (1 + 0.5 * dists2 / self.alpha) ** (-self.alpha)

    def render(self, formatter):
        return f"RQ({formatter(self.alpha)})"

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "RQ"):
        return identical(self.alpha, other.alpha)


@_dispatch
def pairwise(k: RQ, x: B.Numeric, y: B.Numeric):
    return Dense(k._compute(B.pw_dists2(x, y)))


@_dispatch
def elwise(k: RQ, x: B.Numeric, y: B.Numeric):
    return k._compute(B.ew_dists2(x, y))
