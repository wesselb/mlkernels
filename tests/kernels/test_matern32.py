from mlkernels import Linear, Matern32

from ..util import standard_kernel_tests


def test_matern32():
    k = Matern32()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "Matern32()"

    # Test equality.
    assert Matern32() == Matern32()
    assert Matern32() != Linear()

    # Standard tests:
    standard_kernel_tests(k)
