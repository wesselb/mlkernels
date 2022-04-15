import lab as B

from mlkernels import SubspaceKernel, EQ
from ..util import standard_kernel_tests, approx


def test_subspace():
    z = B.randn(3, 2)
    k = SubspaceKernel(EQ(), EQ(), z, EQ()(z))

    # Check parametric type.
    assert type(k) == SubspaceKernel[EQ, EQ]

    # Check that the kernel computes correctly.
    approx(k(z), EQ()(z))

    # Verify that the kernel has the right properties.
    assert not k.stationary
    assert str(k) == f"SubspaceKernel[{EQ}, {EQ}]()"

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
