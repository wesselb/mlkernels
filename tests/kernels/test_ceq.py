from mlkernels import CEQ, Linear

from ..util import standard_kernel_tests


def test_eq():
    k = CEQ(1.0)

    # Verify that the kernel has the right properties.
    assert k.alpha == 1.0
    assert k.stationary
    assert str(k) == "CEQ(1.0)"

    # Test equality.
    assert CEQ(1.0) == CEQ(1.0)
    assert CEQ(1.0) != CEQ(2.0)
    assert CEQ(1.0) != Linear()

    # Standard tests:
    standard_kernel_tests(k)
