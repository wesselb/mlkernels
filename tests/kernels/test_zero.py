import lab as B

from mlkernels import Linear, ZeroKernel

from ..util import approx, standard_kernel_tests


def test_zero():
    k = ZeroKernel()
    x1 = B.randn(10, 2)
    x2 = B.randn(5, 2)

    # Test that the kernel computes correctly.
    approx(k(x1, x2), B.zeros(10, 5))

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "0"

    # Test equality.
    assert ZeroKernel() == ZeroKernel()
    assert ZeroKernel() != Linear()

    # Standard tests:
    standard_kernel_tests(k)
