import lab as B

from mlkernels import EQ, Linear, OneKernel
from ..util import approx, standard_kernel_tests


def test_one():
    k = OneKernel()

    # Check that the kernel computes correctly.
    x1 = B.randn(10, 2)
    x2 = B.randn(5, 2)
    approx(k(x1, x2), B.ones(10, 5))

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "1"

    # Test equality.
    assert OneKernel() == OneKernel()
    assert OneKernel() != Linear()

    # Standard tests:
    standard_kernel_tests(
        k,
        # Test implicit creation of `OneKernel`.
        f1=lambda *xs: (5.0 + EQ())(*xs),
        f2=lambda *xs: 5.0 + EQ()(*xs),
    )
