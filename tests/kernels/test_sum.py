import lab as B
import pytest
from mlkernels import (
    EQ,
    RQ,
    Linear,
    PosteriorKernel,
    ScaledKernel,
    StretchedKernel,
    SubspaceKernel,
    SumKernel,
)

from ..util import standard_kernel_tests


def test_sum():
    k1 = EQ().stretch(2)
    k2 = 3 * RQ(1e-2).stretch(5)
    k = k1 + k2

    # Test parametric type.
    assert type(k) == SumKernel[StretchedKernel[EQ], ScaledKernel[StretchedKernel[RQ]]]

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert EQ() + Linear() == EQ() + Linear()
    assert EQ() + Linear() == Linear() + EQ()
    assert EQ() + Linear() != EQ() + RQ(1e-1)
    assert EQ() + Linear() != RQ(1e-1) + Linear()

    # Standard tests:
    standard_kernel_tests(
        k,
        f1=lambda *xs: (EQ() + RQ(1e-1))(*xs),
        f2=lambda *xs: EQ()(*xs) + RQ(1e-1)(*xs),
    )


z = B.randn(3, 2)
K_z = B.dense(EQ()(z))


@pytest.mark.parametrize(
    "k",
    [
        # Entirely similar:
        PosteriorKernel(EQ(), EQ(), EQ(), z, K_z) + SubspaceKernel(EQ(), EQ(), z, K_z),
        # Dissimilar `z`:
        PosteriorKernel(EQ(), EQ(), EQ(), z, K_z)
        + SubspaceKernel(EQ(), EQ(), z + 1, K_z),
        # Dissimilar `k_zi`:
        PosteriorKernel(EQ(), EQ(), EQ(), z, K_z)
        + SubspaceKernel(2 * EQ(), EQ(), z, K_z),
        # Dissimilar `k_zj`:
        PosteriorKernel(EQ(), EQ(), EQ(), z, K_z)
        + SubspaceKernel(EQ(), 2 * EQ(), z, K_z),
    ],
)
def test_sum_specialisations(k):
    standard_kernel_tests(
        k,
        shapes=[
            ((10, 2), (5, 2)),
            # Add in batch shapes.
            ((3, 10, 2), (3, 5, 2)),
            ((3, 10, 2), (5, 2)),
            ((10, 2), (3, 5, 2)),
        ],
        pd=k[1].k_zi == k[1].k_zj,
    )
