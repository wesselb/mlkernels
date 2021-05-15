import lab as B

from mlkernels import SubspaceKernel, EQ
from ..util import standard_kernel_tests, approx


def test_subspace():
    z = B.randn(3, 2)
    k = SubspaceKernel(EQ(), EQ(), z, EQ()(z))

    # Check that the kernel computes correctly.
    approx(k(z), EQ()(z))

    # Verify that the kernel has the right properties.
    assert not k.stationary
    assert str(k) == "SubspaceKernel()"

    # Standard tests:
    standard_kernel_tests(k, shapes=[((10, 2), (5, 2))])
