from mlkernels import Exp, Linear, Matern12

from ..util import standard_kernel_tests


def test_matern12():
    k = Exp()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "Matern12()"

    # Test equality.
    assert Exp() == Exp()
    assert Exp() == Matern12()
    assert Exp() != Linear()

    # Standard tests:
    standard_kernel_tests(k)
