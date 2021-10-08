from mlkernels import CEQ, Linear

from ..util import standard_kernel_tests


def test_eq():
    k = CEQ()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "CEQ()"

    # Test equality.
    assert CEQ() == CEQ()
    assert CEQ() != Linear()

    # Standard tests:
    standard_kernel_tests(k)
