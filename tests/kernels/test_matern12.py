from mlkernels import Exp, Linear, Matern12

from ..util import standard_kernel_tests


def test_matern12():
    k = Matern12()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "Matern12()"

    # Test equality.
    assert Matern12() == Matern12()
    assert Matern12() == Exp()
    assert Matern12() != Linear()

    # Standard tests:
    standard_kernel_tests(k)
