import lab as B
from algebra import TensorProductFunction
from algebra.util import identical
from matrix import LowRank

from . import _dispatch
from .. import Kernel
from ..util import expand, uprank

__all__ = ["TensorProductKernel"]


class TensorProductKernel(Kernel, TensorProductFunction):
    """Tensor product kernel."""

    @_dispatch
    def __eq__(self, other: "TensorProductKernel"):
        return identical(expand(self.fs), expand(other.fs))


@_dispatch
def pairwise(k: TensorProductKernel, x: B.Numeric, y: B.Numeric):
    f1, f2 = expand(k.fs)
    if x is y and f1 is f2:
        return LowRank(uprank(f1(uprank(x))))
    else:
        return LowRank(left=uprank(f1(uprank(x))), right=uprank(f2(uprank(y))))


@_dispatch
def elwise(k: TensorProductKernel, x: B.Numeric, y: B.Numeric):
    f1, f2 = expand(k.fs)
    if x is y and f1 is f2:
        return B.power(uprank(f1(uprank(x))), 2)
    else:
        return B.multiply(uprank(f1(uprank(x))), uprank(f2(uprank(y))))
