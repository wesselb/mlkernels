import lab as B
from matrix import Dense

from . import _dispatch
from .. import Kernel

__all__ = ["Matern12", "Exp"]


class Matern12(Kernel):
    """Matern--1/2 kernel."""

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "Matern12"):
        return True


Exp = Matern12  #: Alias for the Matern--1/2 kernel.


@_dispatch
def pairwise(k: Matern12, x: B.Numeric, y: B.Numeric):
    return Dense(B.exp(-B.pw_dists(x, y)))


@_dispatch
def elwise(k: Matern12, x: B.Numeric, y: B.Numeric):
    return B.exp(-B.ew_dists(x, y))
