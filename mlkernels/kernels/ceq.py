import lab as B
from matrix import Dense

from . import _dispatch
from .. import Kernel

__all__ = ["CEQ"]


class CEQ(Kernel):
    """Causal exponentiated quadratic kernel."""

    def _compute(self, dists):
        return (1 - B.erf(dists / 4)) * B.exp(-0.5 * dists ** 2)

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "CEQ"):
        return True


@_dispatch
def pairwise(k: CEQ, x: B.Numeric, y: B.Numeric):
    return Dense(k._compute(B.pw_dists(x, y)))


@_dispatch
def elwise(k: CEQ, x: B.Numeric, y: B.Numeric):
    return k._compute(B.ew_dists(x, y))
