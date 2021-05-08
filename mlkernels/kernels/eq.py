import lab as B
from matrix import Dense

from . import _dispatch
from .. import Kernel

__all__ = ["EQ"]


class EQ(Kernel):
    """Exponentiated quadratic kernel."""

    def _compute(self, dists2):
        return B.exp(-0.5 * dists2)

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "EQ"):
        return True


@_dispatch
def pairwise(k: EQ, x: B.Numeric, y: B.Numeric):
    return Dense(k._compute(B.pw_dists2(x, y)))


@_dispatch
def elwise(k: EQ, x: B.Numeric, y: B.Numeric):
    return k._compute(B.ew_dists2(x, y))
