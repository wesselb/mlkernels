import lab as B
from algebra.util import identical
from matrix import Dense

from . import _dispatch
from .. import Kernel

__all__ = ["CEQ"]


class CEQ(Kernel):
    """Causal exponentiated quadratic kernel.

    Args:
        alpha (scalar): Roughness factor.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def _compute(self, dists):
        return (1 - B.erf(self.alpha * dists / 4)) * B.exp(-0.5 * dists**2)

    def render(self, formatter):
        return f"CEQ({formatter(self.alpha)})"

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "CEQ"):
        return identical(self.alpha, other.alpha)


@_dispatch
def pairwise(k: CEQ, x: B.Numeric, y: B.Numeric):
    return Dense(k._compute(B.pw_dists(x, y)))


@_dispatch
def elwise(k: CEQ, x: B.Numeric, y: B.Numeric):
    return k._compute(B.ew_dists(x, y))
