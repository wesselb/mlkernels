import lab as B
from matrix import Dense

from . import _dispatch
from .. import Kernel

__all__ = ["Matern32"]


class Matern32(Kernel):
    """Matern--3/2 kernel."""

    def _compute(self, dists):
        r = 3**0.5 * dists
        return (1 + r) * B.exp(-r)

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "Matern32"):
        return True


@_dispatch
def pairwise(k: Matern32, x: B.Numeric, y: B.Numeric):
    return Dense(k._compute(B.pw_dists(x, y)))


@_dispatch
def elwise(k: Matern32, x: B.Numeric, y: B.Numeric):
    return k._compute(B.ew_dists(x, y))
