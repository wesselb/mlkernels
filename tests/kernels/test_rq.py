from mlkernels import RQ, Linear

from ..util import standard_kernel_tests


def test_rq():
    k = RQ(1e-1)

    # Verify that the kernel has the right properties.
    assert k.alpha == 1e-1
    assert k.stationary
    assert str(k) == "RQ(0.1)"

    # Test equality.
    assert RQ(1e-1) == RQ(1e-1)
    assert RQ(1e-1) != RQ(2e-1)
    assert RQ(1e-1) != Linear()

    # Standard tests:
    standard_kernel_tests(k)
