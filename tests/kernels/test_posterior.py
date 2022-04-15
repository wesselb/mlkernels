import lab as B
import pytest

from mlkernels import PosteriorKernel, EQ
from ..util import standard_kernel_tests, approx


@pytest.mark.parametrize(
    "k_zi, k_zj",
    [
        # Test equal and unequal `k_zi` and `k_zj` to cover all branches.
        (EQ(), EQ()),
        (0.5 * EQ(), 2 * EQ()),
    ],
)
def test_posterior(k_zi, k_zj):
    z = B.randn(3, 2)
    k = PosteriorKernel(k_zi, k_zj, EQ(), z, EQ()(z))

    # Test parametric type.
    assert type(k) == PosteriorKernel[type(k_zi), type(k_zj), EQ]

    # Check that the kernel computes correctly.
    if k_zi == k_zj == k.k_ij:
        approx(k(z), B.zeros(3, 3), atol=1e-11)

    # Verify that the kernel has the right properties.
    assert not k.stationary
    assert str(k) == f"PosteriorKernel[{type(k_zi)}, {type(k_zj)}, {EQ}]()"

    # Standard tests:
    standard_kernel_tests(
        k,
        shapes=[
            ((10, 2), (5, 2)),
            # Add in batch shapes.
            ((3, 10, 2), (3, 5, 2)),
            ((3, 10, 2), (5, 2)),
            ((10, 2), (3, 5, 2)),
        ],
    )
