import lab as B
from algebra import ScaledFunction
from plum import parametric

from . import _dispatch
from .. import Kernel

__all__ = ["ScaledKernel"]


@parametric
class ScaledKernel(Kernel, ScaledFunction):
    """Scaled kernel."""

    @classmethod
    def __infer_type_parameter__(cls, k, *args):
        return type(k)

    @property
    def _stationary(self):
        return self[0].stationary


@_dispatch
def pairwise(k: ScaledKernel, x, y):
    return B.multiply(k.scale, pairwise(k[0], x, y))


@_dispatch
def elwise(k: ScaledKernel, x, y):
    return B.multiply(k.scale, elwise(k[0], x, y))
