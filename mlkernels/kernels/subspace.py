import lab as B
from matrix import AbstractMatrix
from plum import convert, parametric

from . import _dispatch
from .posterior import _K_zi_K_zj
from .. import Kernel

__all__ = ["SubspaceKernel"]


@parametric
class SubspaceKernel(Kernel):
    """Kernel for a subspace of the RKHS.

    Args:
        k_zi (:class:`.kernel.Kernel`): Kernel between the processes corresponding to
            the left input and the inducing points respectively.
        k_zj (:class:`.kernel.Kernel`): Kernel between the processes corresponding to
            the right input and the inducing points respectively.
        z (input): Locations of the inducing points.
        A (matrix): Generalised inducing point kernel matrix.
    """

    @classmethod
    def __infer_type_parameter__(cls, k_zi, k_zj, *args):
        return type(k_zi), type(k_zj)

    def __init__(self, k_zi, k_zj, z, A):
        self.k_zi = k_zi
        self.k_zj = k_zj
        self.z = z
        self.A = convert(A, AbstractMatrix)


@_dispatch
def pairwise(k: SubspaceKernel, x, y):
    return _pairwise_subspacekernel(k, x, y, *_K_zi_K_zj(k.k_zi, k.k_zj, k.z, x, y))


@_dispatch
def _pairwise_subspacekernel(k: SubspaceKernel, x, y, K_zi, K_zj):
    return B.iqf(k.A, K_zi, K_zj)


@_dispatch
def elwise(k: SubspaceKernel, x, y):
    return _elwise_subspacekernel(k, x, y, *_K_zi_K_zj(k.k_zi, k.k_zj, k.z, x, y))


@_dispatch
def _elwise_subspacekernel(k: SubspaceKernel, x, y, K_zi, K_zj):
    return B.iqf_diag(k.A, K_zi, K_zj)[..., :, None]
