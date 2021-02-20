import lab as B
from matrix import Dense
from plum import Dispatcher, Self

from . import _dispatch
from .. import Kernel

__all__ = ["Matern12", "Exp"]


class Matern12(Kernel):
    """Matern--1/2 kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @property
    def _stationary(self):
        return True

    @_dispatch(Self)
    def __eq__(self, other):
        return True


Exp = Matern12  #: Alias for the Matern--1/2 kernel.


@_dispatch(Matern12, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    return Dense(B.exp(-B.pw_dists(x, y)))


@_dispatch(Matern12, B.Numeric, B.Numeric)
def elwise(k, x, y):
    return B.exp(-B.ew_dists(x, y))
