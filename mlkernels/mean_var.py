import lab as B

from . import Mean, Kernel, pairwise, elwise
from . import PosteriorMean, PosteriorKernel, SumKernel, SubspaceKernel
from . import _dispatch
from .kernels.posterior import _pairwise_posteriorkernel, _elwise_posteriorkernel
from .kernels.subspace import _pairwise_subspacekernel, _elwise_subspacekernel

__all__ = ["mean_var", "mean_var_diag"]


@_dispatch
def mean_var(m: Mean, k: Kernel, x):
    """Compute a mean and kernel matrix simultaneously.

    Args:
        m (:class:`.mean.Mean`): Mean function.
        k (:class:`.kernel.Kernel`): Kernel.
        x (input): Input.

    Returns:
        column vector: Mean vector as a rank-2 column vector.
        matrix: Kernel matrix.
    """
    return m(x), pairwise(k, x, x)


@_dispatch
def mean_var_diag(m: Mean, k: Kernel, x):
    """Compute a mean and the diagonal of a kernel matrix simultaneously.

    Args:
        m (:class:`.mean.Mean`): Mean function.
        k (:class:`.kernel.Kernel`): Kernel.
        x (input): Input.

    Returns:
        column vector: Mean vector as a rank-2 column vector.
        column vector: Diagonal of kernel matrix.
    """
    return m(x), elwise(k, x, x)


@_dispatch
def _similar_form(m: PosteriorMean, k: PosteriorKernel):
    same_k = m.k_zi == k.k_zi == k.k_zj
    same_z = m.z is k.z
    return same_k and same_z


@_dispatch
def mean_var(m: PosteriorMean, k: PosteriorKernel, x):
    if not _similar_form(m, k):
        return mean_var.invoke(Mean, Kernel, object)(m, k, x)
    K_zi = m.k_zi(m.z, x)
    return m(x, K_zi), _pairwise_posteriorkernel(k, x, x, K_zi, K_zi)


@_dispatch
def mean_var_diag(m: PosteriorMean, k: PosteriorKernel, x):
    if not _similar_form(m, k):
        return mean_var_diag.invoke(Mean, Kernel, object)(m, k, x)
    K_zi = m.k_zi(m.z, x)
    return m(x, K_zi), _elwise_posteriorkernel(k, x, x, K_zi, K_zi)


@_dispatch
def _similar_form(m: PosteriorMean, k: SumKernel[PosteriorKernel, SubspaceKernel]):
    same_k = m.k_zi == k[0].k_zi == k[0].k_zj == k[1].k_zi == k[1].k_zj
    same_z = m.z is k[0].z is k[1].z
    return same_k and same_z


@_dispatch
def mean_var(
    m: PosteriorMean,
    k: SumKernel[PosteriorKernel, SubspaceKernel],
    x,
):
    if not _similar_form(m, k):
        return mean_var.invoke(Mean, Kernel, object)(m, k, x)
    K_zi = m.k_zi(m.z, x)
    return m(x, K_zi), B.add(
        _pairwise_posteriorkernel(k[0], x, x, K_zi, K_zi),
        _pairwise_subspacekernel(k[1], x, x, K_zi, K_zi),
    )


@_dispatch
def mean_var_diag(
    m: PosteriorMean,
    k: SumKernel[PosteriorKernel, SubspaceKernel],
    x,
):
    if not _similar_form(m, k):
        return mean_var_diag.invoke(Mean, Kernel, object)(m, k, x)
    K_zi = m.k_zi(m.z, x)
    return m(x, K_zi), B.add(
        _elwise_posteriorkernel(k[0], x, x, K_zi, K_zi),
        _elwise_subspacekernel(k[1], x, x, K_zi, K_zi),
    )
