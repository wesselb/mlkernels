from algebra import InputTransformedFunction
from algebra.util import identical
from plum import Dispatcher, Self

from . import _dispatch
from .. import Kernel
from ..util import expand, uprank

__all__ = ["InputTransformedKernel"]


class InputTransformedKernel(Kernel, InputTransformedFunction):
    """Input-transformed kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and identical(expand(self.fs), expand(other.fs))


@_dispatch(InputTransformedKernel, object, object)
def pairwise(k, x, y):
    return pairwise(k[0], *_inputtransformedkernel_compute(k, x, y))


@_dispatch(InputTransformedKernel, object, object)
def elwise(k, x, y):
    return elwise(k[0], *_inputtransformedkernel_compute(k, x, y))


def _inputtransformedkernel_compute(k, x, y):
    f1, f2 = expand(k.fs)
    x = x if f1 is None else f1(uprank(x))
    y = y if f2 is None else f2(uprank(y))
    return x, y
