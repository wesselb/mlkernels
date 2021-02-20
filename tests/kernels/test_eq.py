from mlkernels import EQ, Linear

from ..util import standard_kernel_tests


def test_eq():
    k = EQ()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "EQ()"

    # Test equality.
    assert EQ() == EQ()
    assert EQ() != Linear()

    # Standard tests:
    standard_kernel_tests(k)
