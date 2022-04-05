import lab as B
from algebra.util import identical
from matrix import Dense

from . import _dispatch
from .. import Kernel
from ..util import num_elements

__all__ = ["Delta"]


class Delta(Kernel):
    """Kronecker delta kernel.

    Args:
        epsilon (float, optional): Tolerance for equality in distance.
            Defaults to `1e-6`.
    """

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def _compute(self, dists2):
        dtype = B.dtype(dists2)
        return B.cast(dtype, B.lt(dists2, self.epsilon**2))

    @property
    def _stationary(self):
        return True

    @_dispatch
    def __eq__(self, other: "Delta"):
        return identical(self.epsilon, other.epsilon)


@_dispatch
def pairwise(k: Delta, x: B.Numeric, y: B.Numeric):
    if x is y:
        return B.fill_diag(B.ones(B.dtype(x), *B.shape_batch(x)), num_elements(x))
    else:
        return Dense(k._compute(B.pw_dists2(x, y)))


@_dispatch
def elwise(k: Delta, x: B.Numeric, y: B.Numeric):
    if x is y:
        return B.ones(B.dtype(x), *B.shape_batch(x), num_elements(x), 1)
    else:
        return k._compute(B.ew_dists2(x, y))
