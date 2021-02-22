import lab as B
from algebra.util import identical
from matrix import Dense
from plum import Dispatcher, Self

from . import _dispatch
from .. import Kernel
from ..util import num_elements

__all__ = ["Delta"]


class Delta(Kernel):
    """Kronecker delta kernel.

    Args:
        epsilon (float, optional): Tolerance for equality in squared distance.
            Defaults to `1e-10`.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def _compute(self, dists2):
        dtype = B.dtype(dists2)
        return B.cast(dtype, B.lt(dists2, self.epsilon))

    @property
    def _stationary(self):
        return True

    @_dispatch(Self)
    def __eq__(self, other):
        return identical(self.epsilon, other.epsilon)


@_dispatch(Delta, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    if x is y:
        return B.fill_diag(B.one(x), num_elements(x))
    else:
        return Dense(k._compute(B.pw_dists2(x, y)))


@_dispatch(Delta, B.Numeric, B.Numeric)
def elwise(k, x, y):
    if x is y:
        return B.ones(B.dtype(x), num_elements(x), 1)
    else:
        return k._compute(B.ew_dists2(x, y))
