import lab as B
from algebra import ZeroFunction

from . import _dispatch
from ..mean import Mean
from ..util import num_elements

__all__ = ["ZeroMean"]


class ZeroMean(Mean, ZeroFunction):
    """Constant mean of `0`."""

    @_dispatch
    def __call__(self, x: B.Numeric):
        return B.zeros(B.dtype(x), *B.shape_batch(x), num_elements(x), 1)
