import lab as B
from matrix import Dense

from . import _dispatch
from .. import Kernel

__all__ = ["Matern52"]


class Matern52(Kernel):
    """Matern--5/2 kernel."""

    def _compute(self, dists):
        r1 = 5**0.5 * dists
        r2 = 5 * dists**2 / 3
        return (1 + r1 + r2) * B.exp(-r1)

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "Matern52"):
        return True


@_dispatch
def pairwise(k: Matern52, x: B.Numeric, y: B.Numeric):
    return Dense(k._compute(B.pw_dists(x, y)))


@_dispatch
def elwise(k: Matern52, x: B.Numeric, y: B.Numeric):
    return k._compute(B.ew_dists(x, y))
