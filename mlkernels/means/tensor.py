import lab as B
from algebra import TensorProductFunction

from . import _dispatch
from ..mean import Mean
from ..util import uprank

__all__ = ["TensorProductMean"]


class TensorProductMean(Mean, TensorProductFunction):
    @_dispatch
    def __call__(self, x: B.Numeric):
        return uprank(self.fs[0](uprank(x)))
