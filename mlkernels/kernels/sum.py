import lab as B
from algebra import SumFunction
from plum import parametric

from . import _dispatch
from .posterior import (
    PosteriorKernel,
    _pairwise_posteriorkernel,
    _elwise_posteriorkernel,
    _K_zi_K_zj,
)
from .subspace import (
    SubspaceKernel,
    _pairwise_subspacekernel,
    _elwise_subspacekernel,
)
from .. import Kernel

__all__ = ["SumKernel"]


@parametric
class SumKernel(Kernel, SumFunction):
    """Sum of kernels."""

    @property
    def _stationary(self):
        return self[0].stationary and self[1].stationary


@_dispatch
def pairwise(k: SumKernel, x, y):
    return B.add(pairwise(k[0], x, y), pairwise(k[1], x, y))


@_dispatch
def _similar_form(k: SumKernel[PosteriorKernel, SubspaceKernel]):
    same_k_zi = k[0].k_zi == k[1].k_zi
    same_k_zj = k[0].k_zj == k[1].k_zj
    same_z = k[0].z is k[1].z
    return same_k_zi and same_k_zj and same_z


@_dispatch
def pairwise(k: SumKernel[PosteriorKernel, SubspaceKernel], x, y):
    if _similar_form(k):
        K_zi, K_zj = _K_zi_K_zj(k[0].k_zi, k[0].k_zj, k[0].z, x, y)
        return B.add(
            _pairwise_posteriorkernel(k[0], x, y, K_zi, K_zj),
            _pairwise_subspacekernel(k[1], x, y, K_zi, K_zj),
        )
    return B.add(pairwise(k[0], x, y), pairwise(k[1], x, y))


@_dispatch
def elwise(k: SumKernel, x, y):
    return B.add(elwise(k[0], x, y), elwise(k[1], x, y))


@_dispatch
def elwise(k: SumKernel[PosteriorKernel, SubspaceKernel], x, y):
    if _similar_form(k):
        K_zi, K_zj = _K_zi_K_zj(k[0].k_zi, k[0].k_zj, k[0].z, x, y)
        return B.add(
            _elwise_posteriorkernel(k[0], x, y, K_zi, K_zj),
            _elwise_subspacekernel(k[1], x, y, K_zi, K_zj),
        )
    return B.add(elwise(k[0], x, y), elwise(k[1], x, y))
