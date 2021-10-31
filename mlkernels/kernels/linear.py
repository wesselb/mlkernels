import lab as B
from matrix import LowRank

from . import _dispatch
from .. import Kernel
from ..util import uprank

__all__ = ["Linear"]


class Linear(Kernel):
    """Linear kernel."""

    @property
    def _stationary(self):
        return False

    @_dispatch
    def __eq__(self, other: "Linear"):
        return True


@_dispatch
def pairwise(k: Linear, x: B.Numeric, y: B.Numeric):
    if x is y:
        return LowRank(uprank(x))
    else:
        return LowRank(left=uprank(x), right=uprank(y))


@_dispatch
@uprank
def elwise(k: Linear, x: B.Numeric, y: B.Numeric):
    return B.expand_dims(B.sum(B.multiply(x, y), axis=-1), axis=-1)
