from mlkernels import EQ, Matern12

from ..util import standard_kernel_tests


def test_scaled():
    k = 2 * EQ()

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert 2 * EQ() == 2 * EQ()
    assert 2 * EQ() != 3 * EQ()
    assert 2 * EQ() != 2 * Matern12()

    # Standard tests:
    standard_kernel_tests(k)
