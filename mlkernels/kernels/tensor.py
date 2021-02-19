import lab as B
from algebra import TensorProductFunction
from algebra.util import identical
from matrix import LowRank
from plum import Dispatcher, Self

from . import _dispatch
from .. import Kernel
from ..util import expand, uprank

__all__ = ["TensorProductKernel"]


class TensorProductKernel(Kernel, TensorProductFunction):
    """Tensor product kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(Self)
    def __eq__(self, other):
        return identical(expand(self.fs), expand(other.fs))


@_dispatch(TensorProductKernel, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    f1, f2 = expand(k.fs)
    if x is y and f1 is f2:
        return LowRank(uprank(f1(uprank(x))))
    else:
        return LowRank(left=uprank(f1(uprank(x))), right=uprank(f2(uprank(y))))


@_dispatch(B.Numeric, B.Numeric)
def elwise(k, x, y):
    f1, f2 = expand(k.fs)
    if x is y and f1 is f2:
        return B.power(uprank(f1(uprank(x))), 2)
    else:
        return B.multiply(uprank(f1(uprank(x))), uprank(f2(uprank(y))))
