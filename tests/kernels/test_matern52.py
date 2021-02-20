from mlkernels import Linear, Matern52

from ..util import standard_kernel_tests


def test_matern52():
    k = Matern52()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "Matern52()"

    # Test equality.
    assert Matern52() == Matern52()
    assert Matern52() != Linear()

    # Standard tests:
    standard_kernel_tests(k)
