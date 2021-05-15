import lab as B
from matrix import AbstractMatrix
from plum import convert

from . import _dispatch
from .. import Kernel

__all__ = ["SubspaceKernel"]


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

    def __init__(self, k_zi, k_zj, z, A):
        self.k_zi = k_zi
        self.k_zj = k_zj
        self.z = z
        self.A = convert(A, AbstractMatrix)


@_dispatch
def pairwise(k: SubspaceKernel, x, y):
    return B.iqf(k.A, k.k_zi(k.z, x), k.k_zj(k.z, y))


@_dispatch
def elwise(k: SubspaceKernel, x, y):
    return B.iqf_diag(k.A, k.k_zi(k.z, x), k.k_zj(k.z, y))[:, None]
