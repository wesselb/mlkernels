from mlkernels import EQ, Linear

from ..util import standard_kernel_tests


def test_linear():
    k = Linear()

    # Verify that the kernel has the right properties.
    assert not k.stationary
    assert str(k) == "Linear()"

    # Test equality.
    assert Linear() == Linear()
    assert Linear() != EQ()

    # Standard tests:
    standard_kernel_tests(k)
