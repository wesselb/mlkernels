import lab as B
from matrix import LowRank
from plum import Dispatcher, Self

from . import _dispatch
from .. import Kernel
from ..util import uprank

__all__ = ["Linear"]


class Linear(Kernel):
    """Linear kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @property
    def _stationary(self):
        return False

    @_dispatch(Self)
    def __eq__(self, other):
        return True


@_dispatch(Linear, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    if x is y:
        return LowRank(uprank(x))
    else:
        return LowRank(left=uprank(x), right=uprank(y))


@_dispatch(Linear, B.Numeric, B.Numeric)
@uprank
def elwise(k, x, y):
    return B.expand_dims(B.sum(B.multiply(x, y), axis=1), axis=1)
