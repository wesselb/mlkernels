import lab as B
from algebra import OneFunction

from . import _dispatch
from ..mean import Mean
from ..util import num_elements

__all__ = ["OneMean"]


class OneMean(Mean, OneFunction):
    """Constant mean of `1`."""

    @_dispatch
    def __call__(self, x: B.Numeric):
        return B.ones(B.dtype(x), num_elements(x), 1)
