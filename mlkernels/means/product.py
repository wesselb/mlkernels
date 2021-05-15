import lab as B
from algebra import ProductFunction

from . import _dispatch
from ..mean import Mean

__all__ = ["ProductMean"]


class ProductMean(Mean, ProductFunction):
    """Product of two means."""

    @_dispatch
    def __call__(self, x):
        return B.multiply(self[0](x), self[1](x))
