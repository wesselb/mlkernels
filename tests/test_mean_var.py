import lab as B
import pytest

from mlkernels import (
    Mean,
    Kernel,
    PosteriorMean,
    PosteriorKernel,
    SubspaceKernel,
    EQ,
    Exp,
    TensorProductMean,
)
from mlkernels.mean_var import mean_var, mean_var_diag
from .util import approx

z = B.linspace(0, 5, 5)
z2 = B.linspace(-5, 0, 5)
y = B.randn(5, 1)

m_i = TensorProductMean(lambda x: x**2)
m_z = TensorProductMean(lambda x: x)

candidates = [
    # No optimisation:
    (m_i, EQ()),
    # Posterior kernel computation:
    (
        PosteriorMean(m_i, m_z, 0.5 * EQ(), z, 0.25 * EQ()(z), y),
        PosteriorKernel(EQ(), 0.5 * EQ(), 0.5 * EQ(), z, 0.25 * EQ()(z)),
    ),
    # Posterior cross-kernel computation:
    (
        PosteriorMean(m_i, m_z, 0.5 * EQ(), z, 0.25 * EQ()(z), y),
        PosteriorKernel(EQ(), EQ(), 0.5 * EQ(), z, 0.25 * EQ()(z)),
    ),
    # Decoupled posterior kernel computation:
    (
        PosteriorMean(m_i, m_z, 0.5 * Exp(), z2, 0.25 * Exp()(z2), y),
        PosteriorKernel(EQ(), 0.5 * EQ(), 0.5 * EQ(), z, 0.25 * EQ()(z)),
    ),
    # Decoupled posterior cross-kernel computation:
    (
        PosteriorMean(m_i, m_z, 0.5 * Exp(), z, 0.25 * Exp()(z), y),
        PosteriorKernel(EQ(), EQ(), 0.5 * EQ(), z, 0.25 * EQ()(z)),
    ),
    # Pseudo-posterior kernel computation:
    (
        PosteriorMean(m_i, m_z, 0.5 * EQ(), z, 0.25 * EQ()(z), y),
        PosteriorKernel(EQ(), 0.5 * EQ(), 0.5 * EQ(), z, 0.25 * EQ()(z))
        + SubspaceKernel(0.5 * EQ(), 0.5 * EQ(), z, B.eye(5)),
    ),
    # Pseudo-posterior cross-kernel computation:
    (
        PosteriorMean(m_i, m_z, 0.5 * EQ(), z, 0.25 * EQ()(z), y),
        PosteriorKernel(EQ(), EQ(), 0.5 * EQ(), z, 0.25 * EQ()(z))
        + SubspaceKernel(EQ(), 0.5 * EQ(), z, B.eye(5)),
    ),
    # Decoupled pseudo-posterior kernel computation:
    (
        PosteriorMean(m_i, m_z, 0.5 * Exp(), z2, 0.25 * Exp()(z2), y),
        PosteriorKernel(EQ(), 0.5 * EQ(), 0.5 * EQ(), z, 0.25 * EQ()(z))
        + SubspaceKernel(0.5 * EQ(), 0.5 * EQ(), z, B.eye(5)),
    ),
    # Decoupled pseudo-posterior cross-kernel computation:
    (
        PosteriorMean(m_i, m_z, 0.5 * Exp(), z, 0.25 * Exp()(z), y),
        PosteriorKernel(EQ(), EQ(), 0.5 * EQ(), z, 0.25 * EQ()(z))
        + SubspaceKernel(EQ(), 0.5 * EQ(), z, B.eye(5)),
    ),
]


@pytest.mark.parametrize("m, k", candidates)
def test_mean_var(m, k):
    x = B.linspace(2.5, 7.5, 10)
    approx(mean_var(m, k, x), mean_var.invoke(Mean, Kernel, object)(m, k, x))


@pytest.mark.parametrize("m, k", candidates)
def test_mean_var_diag(m, k):
    x = B.linspace(2.5, 7.5, 10)
    approx(mean_var_diag(m, k, x), mean_var_diag.invoke(Mean, Kernel, object)(m, k, x))
