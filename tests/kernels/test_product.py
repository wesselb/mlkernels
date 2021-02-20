from mlkernels import EQ, RQ, Linear

from ..util import standard_kernel_tests


def test_product():
    k = (2 * EQ().stretch(10)) * (3 * RQ(1e-2).stretch(20))

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert EQ() * Linear() == EQ() * Linear()
    assert EQ() * Linear() == Linear() * EQ()
    assert EQ() * Linear() != EQ() * RQ(1e-1)
    assert EQ() * Linear() != RQ(1e-1) * Linear()

    # Standard tests:
    standard_kernel_tests(k)
